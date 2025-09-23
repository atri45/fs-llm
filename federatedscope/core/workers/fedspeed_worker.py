import shutil
import time
import torch
import psutil
import os
import logging
import torch.distributed as dist
import pickle
import base64
import numpy as np

from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, add_prefix_to_path
from federatedscope.llm.trainer.fedspeed_engine import FedSpeedCoordinatorEngine
from federatedscope.llm.trainer.fedspeed_trainer import FedSpeedTrainer, EarlyStopException
from torch.nn.utils.convert_parameters import vector_to_parameters
from collections import OrderedDict
from federatedscope.core.communication import gRPCCommManager
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, device):
        self.device = device
        self.start_time = 0.0
        self.end_time = 0.0
        
        # 追踪 CPU 内存 (RAM)
        self.process = psutil.Process(os.getpid())
        self.cpu_mem_usage_peak = 0.0 # 单位: MB

        # 追踪 GPU 显存 (VRAM)
        self.gpu_mem_allocated_peak = 0.0 # 单位: MB
        self.gpu_mem_reserved_peak = 0.0  # 单位: MB

    def _is_cuda(self):
        """一个辅助函数，用于判断是否在使用 CUDA。"""
        if not torch.cuda.is_available():
            return False
        if isinstance(self.device, torch.device):
            return self.device.type == 'cuda'
        elif isinstance(self.device, str):
            return 'cuda' in self.device
        elif isinstance(self.device, int):
            return self.device >= 0 # 假设非负整数代表 GPU ID
        return False

    def start(self):
        """开始计时和监控。"""
        # 使用辅助函数进行判断
        if self._is_cuda():
            torch.cuda.reset_peak_memory_stats(self.device)
        
        self.start_time = time.time()
        logger.info("Performance monitor started.")
        
    def stop(self):
        """停止计时，收集峰值数据，并打印报告。"""
        self.end_time = time.time()
        
        # 收集峰值数据
        self.cpu_mem_usage_peak = self.process.memory_info().rss / (1024 ** 2)
        
        if self._is_cuda():
            stats = torch.cuda.memory_stats(self.device)
            self.gpu_mem_allocated_peak = stats["allocated_bytes.all.peak"] / (1024 ** 2)
            self.gpu_mem_reserved_peak = stats["reserved_bytes.all.peak"] / (1024 ** 2)

        self.report()

    def report(self):
        """打印性能报告。"""
        total_seconds = self.end_time - self.start_time
        total_minutes = total_seconds / 60.0

        logger.info("----------- Performance Report -----------")
        logger.info(f"  - Total Training Time: {total_minutes:.2f} minutes ({total_seconds:.2f} seconds)")
        logger.info(f"  - CPU Memory Peak Usage (RSS): {self.cpu_mem_usage_peak:.2f} MB")
        
        if self._is_cuda():
            logger.info(f"  - GPU Memory Peak Allocated: {self.gpu_mem_allocated_peak:.2f} MB")
            logger.info(f"  - GPU Memory Peak Reserved: {self.gpu_mem_reserved_peak:.2f} MB")
        else:
            logger.info("  - GPU Monitoring: Not available (CUDA not found or not used).")
        logger.info("------------------------------------------")


class FedSpeedServer(Server):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fedspeed_ready_client_count = 0
        self.world_size = 0
        self.fedspeed_participants = []
        self.coordinator_engine = None
        self.final_shards_buffer = {}
        self.rank_to_client_id_map = {}
        
        self.asynchronous = self._cfg.federate.get('asynchronous_aggregation', False)
        self.anonymous_routing = self._cfg.federate.get('anonymous_routing', False)
        if self._cfg.federate.aggregation_mode == 'shardedGradient':
            self.anonymous_routing = True
        if self._cfg.federate.aggregation_mode == 'fl-sim':
            self.asynchronous = False

        self.register_handlers('fedspeed_ready', self.callback_funcs_for_fedspeed_ready)
        if self.asynchronous == False:
            #  为同步聚合准备缓冲区
            self.sync_model_buffer = {}
            self.register_handlers('sync_model_para', self.callback_funcs_for_sync_model_para)
        if self.anonymous_routing:
            self.public_key_buffer = {}

    def trigger_for_start(self):
        if self.check_client_join_in():
            # 只有在 fedspeed 模式下，才执行特殊的建组逻辑
            if self._cfg.federate.method.lower() == 'fedspeed':
                logger.info("FedSpeed mode detected. Starting distributed group setup...")
                self.fedspeed_setup()
                return # 提前返回，不执行常规的广播
            else:
                # 如果不是 fedspeed 模式，就执行父类的原始逻辑
                super().trigger_for_start()
    
    def fedspeed_setup(self):
        participants = list(self.comm_manager.neighbors.keys())
        self.world_size = len(participants)
        self.fedspeed_participants = sorted(list(self.comm_manager.neighbors.keys()))

        if self.world_size < 2: # 至少需要2个参与方
            logger.error(f"FedSpeed requires at least 2 clients, but only {self.world_size} joined.")
            self.terminate()
            return

        self.rank_to_client_id_map = {i: client_id for i, client_id in enumerate(self.fedspeed_participants)}
        master_addr = self._cfg.distribute.server_host
        master_port = self._cfg.distribute.server_port + 100
        
        logger.info(f"FedSpeed Coordinator: master_addr={master_addr}, master_port={master_port}, world_size={self.world_size}")

        for i, client_id in enumerate(participants):
            rank = i
            dist_config = {
                'master_addr': master_addr,
                'master_port': master_port,
                'world_size': self.world_size,
                'rank': rank,
                'client_id': client_id
            }
            
            self.comm_manager.send(
                Message(msg_type='fedspeed_setup',
                        sender=self.ID,
                        receiver=[client_id],
                        state=self.state,
                        content=dist_config)
            )

    def callback_funcs_for_fedspeed_ready(self, message: Message):
        """
        当所有客户端准备好后，进行密钥交换和分发初始化包。
        """
        self.fedspeed_ready_client_count += 1
        logger.info(f"FedSpeed: Received 'ready' from Client #{message.sender}. ({self.fedspeed_ready_client_count}/{len(self.fedspeed_participants)})")
        if self.anonymous_routing:
            sender_id = message.sender
            # content 现在是 PEM 格式的字节串，可以直接存储
            public_key_b64_str = message.content
            self.public_key_buffer[sender_id] = public_key_b64_str
        
        if self.fedspeed_ready_client_count == len(self.fedspeed_participants):
            logger.info("All clients are ready! Server is now preparing initialization packages...")
            self.start_training_package_distribution()

    def start_training_package_distribution(self):
        """
        在服务器端进行分片和演练，然后分发初始化包。
        """
        # 1. 创建并运行协调引擎
        self.coordinator_engine = FedSpeedCoordinatorEngine(self.model, self._cfg, self.world_size)
        all_client_packages = self.coordinator_engine.profile_and_shard()
        self.coordinator_engine = None

        # 2. 向每个客户端发送其专属的初始化包
        for i, client_id in enumerate(self.fedspeed_participants):
            package_to_send = all_client_packages[i]
            package_to_send['rank_to_client_id_map'] = self.rank_to_client_id_map
            address_book = self.comm_manager.get_neighbors()
            package_to_send['address_book'] = address_book
            if self.anonymous_routing:
                package_to_send['public_key_book'] = self.public_key_buffer

            # 序列化并发送
            content = base64.b64encode(pickle.dumps(package_to_send)).decode('ascii')
            
            self.comm_manager.send(
                Message(msg_type='fedspeed_init_package',
                        sender=self.ID,
                        receiver=[client_id],
                        state=self.state,
                        content=content)
            )
        logger.info("All initialization packages have been sent to clients. Start training……")

    def callback_funcs_for_sync_model_para(self, message: Message):
        """
        处理客户端在同步模式下上传的模型参数。
        """
        sender_id = message.sender
        current_step = message.state
        content = message.content
        
        # bytes 转为 tensor
        try:
            pickled_bytes = base64.b64decode(content.encode('ascii'))
            # unpickle 后的对象是 CPU Tensor，需要转为 numpy
            params_vec = pickle.loads(pickled_bytes).numpy()
        except Exception as e:
            logger.error(f"Failed to deserialize b64_pickled_tensor from client #{sender_id}: {e}")
            return

        # 将收到的模型存入当前 step 的缓冲区
        if current_step not in self.sync_model_buffer:
            self.sync_model_buffer[current_step] = {}
        self.sync_model_buffer[current_step][sender_id] = params_vec
        
        logger.debug(f"Server: Received sync model from Client #{sender_id} for Step #{current_step}. "
                     f"({len(self.sync_model_buffer[current_step])}/{self.world_size})")

        # 检查是否收齐了所有客户端的模型
        if len(self.sync_model_buffer[current_step]) == self.world_size:
            logger.debug(f"Server: All models for Step #{current_step} received. Aggregating and broadcasting...")
            
            # 执行聚合
            all_vectors = list(self.sync_model_buffer[current_step].values())
            try:
                # 将 list of lists 转换为 2D NumPy array,这假设所有 list 的长度都相同
                numpy_array_2d = np.array(all_vectors, dtype=np.float32)
                
                # 调用 .mean(axis=0) 高效地计算平均值
                aggregated_numpy_array = numpy_array_2d.mean(axis=0)
            
                # 将 numpy array 转为 CPU Tensor
                agg_tensor = torch.from_numpy(aggregated_numpy_array)
                pickled_bytes = pickle.dumps(agg_tensor)
                content_to_broadcast = base64.b64encode(pickled_bytes).decode('ascii')

                # 将聚合后的新模型广播给所有客户端
                self.comm_manager.send(
                    Message(msg_type='aggregated_model_para',
                            sender=self.ID,
                            receiver=list(self.comm_manager.neighbors.keys()),
                            state=current_step,
                            content=content_to_broadcast)
                )
            
            except Exception as e:
                logger.error(f"Error during aggregation for Step #{current_step}: {e}", exc_info=True)
                # 清理缓冲区并返回，避免卡死
                del self.sync_model_buffer[current_step]
                return

            # 清理该 step 的缓冲区
            del self.sync_model_buffer[current_step]
            

class FedSpeedClient(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_rank = None
        self.dist_group_initialized = False
        self.monitor = PerformanceMonitor(self.device)
        self.asynchronous = self._cfg.federate.get('asynchronous_aggregation', False)
        self.anonymous_routing = self._cfg.federate.get('anonymous_routing', False)
        self.is_decentralized_sharded_mode = self._cfg.federate.aggregation_mode == 'shardedGradient'
        if self.is_decentralized_sharded_mode:
            self.anonymous_routing = True
        if self._cfg.federate.aggregation_mode == 'fl-sim':
            self.asynchronous = False
        
        self.register_handlers('fedspeed_setup', self.callback_funcs_for_fedspeed_setup)
        self.register_handlers('fedspeed_init_package', self.callback_funcs_for_init_package)

        if self.asynchronous:
            # 所有异步模式都需要处理独立的投票消息
            self.register_handlers('early_stop_vote', self.trainer.passive_receive_message)

            # 分模式注册数据消息处理器
            if self.is_decentralized_sharded_mode:
                self.register_handlers('anonymous_forward', self.trainer.passive_receive_message)
                self.register_handlers('parameter_shard_update', self.trainer.passive_receive_message)
            else:
                # 其他异步模式（如 'model', 'local-reconstruct' 等）的消息处理器
                self.register_handlers('async_model_para', self.trainer.passive_receive_message)
                # 如果也支持匿名路由，则添加
                if self.anonymous_routing:
                    self.register_handlers('anonymous_forward', self.trainer.passive_receive_message)
        else:
            # 同步模式的处理器
            self.register_handlers('aggregated_model_para', self.trainer.handle_model)

    def callback_funcs_for_fedspeed_setup(self, message: Message):
        if dist.is_initialized():
            logger.warning(f"Client #{self.ID}: Default process group already initialized. Destroying it before creating new FedSpeed group.")
            # 销毁已经存在的默认通信组
            dist.destroy_process_group()

        if self.dist_group_initialized:
            logger.warning(f"Client #{self.ID} received fedspeed_setup, but already initialized.")
            return

        config = message.content
        self.my_rank = config['rank']

        if 'client_id' in config:
            self.ID = config['client_id']
            logger.info(f"Client ID has been set to #{self.ID}.")

        # ... (完整的 init_process_group 逻辑) ...
        try:
            os.environ['MASTER_ADDR'] = config['master_addr']
            os.environ['MASTER_PORT'] = str(config['master_port'])
            
            dist.init_process_group(
                backend='gloo',
                rank=config['rank'],
                world_size=config['world_size']
            )
            self.dist_group_initialized = True
            logger.info(f"Client #{self.ID} (Rank {config['rank']}) successfully joined group.")

            # 如果需要匿名路由，则立即初始化 Router 并发送公钥
            public_key_pem_bytes = None
            if self.anonymous_routing:
                 # a. 确保 Trainer 存在
                if not isinstance(self.trainer, FedSpeedTrainer): 
                    raise TypeError(f"FedSpeedClient is not configured with FedSpeedTrainer.")
                
                # b. 手动触发 Trainer 的 Router 初始化
                self.trainer.initialize_router(self.ID, self.comm_manager)
                
                # c. 发送公钥
                if hasattr(self.trainer, 'router') and self.trainer.router:
                    logger.info(f"Client #{self.ID}: Sending public key to server.")
                    public_key_pem_bytes = self.trainer.router.public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    )
                # 将 PEM 字节串编码为 Base64 字符串
                public_key_b64_str = base64.b64encode(public_key_pem_bytes).decode('ascii')

                self.comm_manager.send(
                    Message(msg_type='fedspeed_ready',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content=public_key_b64_str)
                )
                return

            # 发送 ready 信号
            self.comm_manager.send(
                Message(msg_type='fedspeed_ready',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state)
            )

        except Exception as e:
            logger.error(f"Client #{self.ID} failed to join group: {e}")

    def callback_funcs_for_init_package(self, message: Message):
        """
        接收初始化包，并启动基于 Step 的训练-评估循环。
        """
        logger.info(f"Client #{self.ID} (Rank {self.my_rank}): Received init package. Starting step-based run.")

        try:
            # 1. 初始化 Trainer 和 Engine
            init_package = pickle.loads(base64.b64decode(message.content))

            if not isinstance(self.trainer, FedSpeedTrainer):
                raise TypeError(f"FedSpeedClient is not configured with FedSpeedTrainer.")
            self.trainer.fedspeed_init_package = init_package

            # 2. 获取并更新 Trainer 的 peers 列表
            address_book = init_package.get('address_book')
            if address_book:
                logger.info(f"Client #{self.ID}: Received address book. Populating neighbors.")
                for client_id, address in address_book.items():
                    if client_id != self.ID:
                        self.comm_manager.add_neighbors(client_id, address)
            self.trainer.comm_manager = self.comm_manager
            self.trainer.client_id = self.ID
            all_neighbors = list(self.comm_manager.neighbors.keys())
            # 服务器 ID 通常是 0
            peers = [cid for cid in all_neighbors if cid != self.ID and cid != 0]
            self.trainer.peers = peers

            # 3. 配置公钥通讯录
            if self.anonymous_routing:
                public_key_book_serialized = init_package.get('public_key_book')
                if hasattr(self.trainer, 'router') and self.trainer.router and public_key_book_serialized:
                    # 遍历收到的序列化公钥
                    for client_id, key_b64_str in public_key_book_serialized.items():
                        if client_id == self.ID: continue
                        try:
                            # 将 Base64 字符串解码回 PEM 字节串
                            key_pem_bytes = base64.b64decode(key_b64_str)
                            
                            # 使用字节串反序列化为对象
                            public_key_obj = serialization.load_pem_public_key(
                                key_pem_bytes
                            )
                            # 存入 Router
                            self.trainer.router.public_keys[client_id] = public_key_obj
                        except Exception as e:
                            logger.error(f"Failed to deserialize public key for Client #{client_id}: {e}")
                    # Router 现在拥有了所有 peers 的公钥
                    # 它也需要知道 peers 的 ID
                    self.trainer.router.peers = self.trainer.peers
                    self.trainer.router.all_client_ids = [self.trainer.client_id] + self.trainer.peers
                    logger.info(f"Client #{self.ID}: Public key book received and configured.")
                else:
                    raise RuntimeError("Anonymous routing enabled, but public key book not received or router not initialized.")

            # 4. 现在可以安全地开始训练了
            self.monitor.start()
            logger.info("All setup complete. Starting FedSpeed training...")
            self.trainer.train()

        except EarlyStopException:
            logger.info(f"Client #{self.ID}: Early stopping signal received.")

        finally:
            # 4. 无论训练是否成功，都尝试上传最终结果
            # 这确保了即使训练中途出错，也能保存当时的模型状态
            logger.info("Training finished or interrupted.")
            self.monitor.stop()
            # if self.trainer.early_stopper and self.trainer.early_stopper.early_stop:
            #     self.trainer.early_stopper.load_best_checkpoint(self.trainer.fedspeed_engine)
            self.save_final_model_locally()
            self.comm_manager.stop()

    def save_final_model_locally(self):
        """
        将最终的【完整】可训练参数保存到本地磁盘。
        """
        if not (hasattr(self.trainer, 'fedspeed_engine') and self.trainer.fedspeed_engine):
            logger.error("FedSpeedEngine not found. Cannot save model.")
            return

        engine = self.trainer.fedspeed_engine
        # 获取 EarlyStopper 保存的临时检查点文件路径
        best_model_checkpoint_path = self.trainer.early_stopper.checkpoint_path
        
        if not os.path.exists(best_model_checkpoint_path):
            logger.error(f"Best model checkpoint file not found at '{best_model_checkpoint_path}'. Cannot save final model.")
            return

        # --- 最终文件保存的目标路径 (与你之前的逻辑一致) ---
        save_dir = self._cfg.federate.get('client_save_path', './fedspeed_final_models/')
        os.makedirs(save_dir, exist_ok=True)
        
        base_filename = self._cfg.federate.save_to
        final_filename = f"client_{self.ID}_{base_filename}"
        final_save_path = os.path.join(save_dir, final_filename)

        try:
            # 使用 shutil.move 来重命名并移动文件，更高效
            shutil.move(best_model_checkpoint_path, final_save_path)
            
            best_score_info = ""
            if self.trainer.early_stopper.best_score is not None:
                best_score_info = f"Best eval loss: {self.trainer.early_stopper.best_score:.4f}. "
            
            logger.info(f"Client #{self.ID} (Rank 0): {best_score_info}Final best model saved to '{final_save_path}'")

        except Exception as e:
            logger.error(f"Failed to move best model checkpoint from '{best_model_checkpoint_path}' to '{final_save_path}': {e}")  