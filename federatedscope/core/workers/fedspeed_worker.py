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
from federatedscope.llm.trainer.fedspeed_trainer import AggregationTimer, FedSpeedTrainer, EarlyStopException
from torch.nn.utils.convert_parameters import vector_to_parameters
from collections import OrderedDict
from federatedscope.core.communication import gRPCCommManager
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)


class CommunicationMonitor:
    """
    一个简单的类，用于追踪客户端在【训练过程中】发送的总数据量。
    """
    def __init__(self, client_id):
        self.client_id = client_id
        self._total_bytes_sent = 0
        self.gb_factor = 1024 ** 3
        self.is_monitoring = False

    def start(self):
        """开始监控。"""
        logger.debug(f"Communication monitor for Client #{self.client_id} started.")
        self._total_bytes_sent = 0 # 每次开始时重置计数器
        self.is_monitoring = True

    def stop(self):
        """停止监控。"""
        logger.debug(f"Communication monitor for Client #{self.client_id} stopped.")
        self.is_monitoring = False

    def record_sent_data(self, message_obj):
        """
        记录一次发送的数据大小。
        """
        if not self.is_monitoring: # 只有在监控状态下才记录
            return
        
        if message_obj is None or message_obj.content is None:
            return
        
        try:
            payload_bytes = pickle.dumps(message_obj.content)
            self._total_bytes_sent += len(payload_bytes)
        except (pickle.PicklingError, TypeError) as e:
            logger.warning(f"Could not estimate size of message content (type: {type(message_obj.content)}). Skipping. Error: {e}")

    def record_dist_communication(self, op_name: str, tensor: torch.Tensor, world_size: int):
        """
        记录一次 torch.distributed 操作产生的通信量。
        """
        # --- START OF MODIFICATION ---
        if not self.is_monitoring: # <-- 新增：只有在监控状态下才记录
            return
        # --- END OF MODIFICATION ---
        
        if not isinstance(tensor, torch.Tensor):
            return

        tensor_bytes = tensor.numel() * tensor.element_size()
        
        if op_name in ['all_gather', 'all_reduce', 'reduce_scatter']:
            communication_bytes = tensor_bytes * (world_size - 1)
            self._total_bytes_sent += communication_bytes

    @property
    def total_gb_sent(self):
        """返回以GB为单位的总发送量。"""
        return self._total_bytes_sent / self.gb_factor

    def report(self):
        """打印最终的通信成本报告。"""
        logger.info(f"----------- Client #{self.client_id} Communication Report -----------")
        logger.info(f"  - Total Data Sent: {self.total_gb_sent:.4f} GB")
        logger.info(f"  - Total Data Sent (Bytes): {self._total_bytes_sent} bytes")
        logger.info("----------------------------------------------------")


class FedSpeedServer(Server):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fedspeed_ready_client_count = 0
        self.world_size = 0
        self.fedspeed_participants = []
        self.coordinator_engine = None
        self.final_shards_buffer = {}
        self.rank_to_client_id_map = {}
        self.comm_monitor = CommunicationMonitor(self.ID) 
        self._original_send = self.comm_manager.send
        self.comm_manager.send = self._send_with_monitoring
        self.aggregation_timer = AggregationTimer()
        
        self.asynchronous = self._cfg.federate.get('asynchronous_aggregation', False)
        self.anonymous_routing = self._cfg.federate.get('anonymous_routing', False)
        if self._cfg.federate.aggregation_mode == 'shardedGradient':
            self.anonymous_routing = True
        if self._cfg.federate.aggregation_mode == 'fl-sim':
            self.asynchronous = False

        self.register_handlers('fedspeed_ready', self.callback_funcs_for_fedspeed_ready)
        self.register_handlers('finish', self.terminate)
        if self.asynchronous == False:
            #  为同步聚合准备缓冲区
            self.sync_model_buffer = {}
            self.register_handlers('sync_model_para', self.callback_funcs_for_sync_model_para)
        if self.anonymous_routing:
            self.public_key_buffer = {}

    def _send_with_monitoring(self, message: Message):
        """
        一个新的 send 方法，它在调用原始 send 方法之前记录数据大小。
        """
        # 记录数据大小
        # 注意：广播消息会被记录多次，我们需要正确处理
        
        # 检查接收者是不是一个列表（广播或多播）
        num_receivers = 1
        if isinstance(message.receiver, list):
            num_receivers = len(message.receiver)
        
        # 估算单个消息的大小
        payload_bytes = 0
        if message.content is not None:
            try:
                payload_bytes = len(pickle.dumps(message.content))
            except (pickle.PicklingError, TypeError):
                pass # 忽略无法序列化的内容

        # 总发送量 = 单个消息大小 * 接收者数量
        total_bytes_sent_this_call = payload_bytes * num_receivers
        
        # 只有在监控状态下才累加
        if self.comm_monitor.is_monitoring:
            self.comm_monitor._total_bytes_sent += total_bytes_sent_this_call
        
        # 调用原始的 send 方法完成实际的发送
        return self._original_send(message)

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
            self.comm_monitor.start()

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
        self.aggregation_timer.start()
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
            self.aggregation_timer.stop()
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
                self.aggregation_timer.stop()
                return

            # 清理该 step 的缓冲区
            del self.sync_model_buffer[current_step]
            self.aggregation_timer.stop()
            
    def terminate(self, message: Message):
        """
        在服务器终止时，停止监控并打印报告。
        """
        # 停止监控并打印报告
        self.aggregation_timer.report()
        if hasattr(self, 'comm_monitor'):
            self.comm_monitor.stop()
            self.comm_monitor.report()

class FedSpeedClient(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_rank = None
        self.dist_group_initialized = False
        self.asynchronous = self._cfg.federate.get('asynchronous_aggregation', False)
        self.anonymous_routing = self._cfg.federate.get('anonymous_routing', False)
        self.is_decentralized_sharded_mode = self._cfg.federate.aggregation_mode == 'shardedGradient'
        if self.is_decentralized_sharded_mode:
            self.anonymous_routing = True
        if self._cfg.federate.aggregation_mode == 'fl-sim':
            self.asynchronous = False

        self.comm_monitor = CommunicationMonitor(self.ID)
        # 包裹comm_manager.send 方法
        # 保存原始的 send 方法
        self._original_send = self.comm_manager.send
        # 用带监控功能的 send 方法替换它
        self.comm_manager.send = self._send_with_monitoring

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

    def _send_with_monitoring(self, message: Message):
        """
        一个新的 send 方法，它在调用原始 send 方法之前记录数据大小。
        """
        # 记录数据大小
        self.comm_monitor.record_sent_data(message)
        # 调用原始的 send 方法完成实际的发送
        return self._original_send(message)

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
            self.comm_monitor.client_id = self.ID
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
                self.trainer.initialize_router(self.ID)
                
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
            # self.monitor.start()
            logger.info("All setup complete. Starting FedSpeed training...")
            self.comm_monitor.start()
            self.trainer.train()

        except EarlyStopException:
            logger.info(f"Client #{self.ID}: Early stopping signal received.")

        finally:
            # 4. 无论训练是否成功，都尝试上传最终结果
            # 这确保了即使训练中途出错，也能保存当时的模型状态
            self.comm_monitor.stop()
            self.trainer.monitor.stop()
            logger.info("Training finished or interrupted.")
            self.comm_monitor.report() 
            self.save_final_model_locally()
            # 发送训练结束信号给服务器
            try:
                self.comm_manager.send(
                    Message(msg_type='finish',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            content="done") # content 可以是任意内容
                )
                # 等待一小段时间，确保消息有足够的时间被发送出去
                time.sleep(2)
            except Exception as e:
                logger.error(f"Client #{self.ID}: Failed to send 'finish' message to server. Error: {e}")
            self.comm_manager.stop()

    def save_final_model_locally(self):
        """
        将最终的最佳模型以 FederatedScope 兼容的、完整的 state_dict 格式保存到本地磁盘。
        """
        if self.my_rank != 0:
            logger.info(f"Client #{self.ID} (Rank {self.my_rank}): Not Rank 0, skipping final model save.")
            # 即使不保存，也应该清理临时文件
            best_model_checkpoint_path = self.trainer.early_stopper.checkpoint_path
            if os.path.exists(best_model_checkpoint_path):
                os.remove(best_model_checkpoint_path)
            return
        
        if not (hasattr(self.trainer, 'fedspeed_engine') and self.trainer.fedspeed_engine):
            logger.error("FedSpeedEngine not found. Cannot save model.")
            return

        engine = self.trainer.fedspeed_engine
        best_model_checkpoint_path = self.trainer.early_stopper.checkpoint_path
        
        if not os.path.exists(best_model_checkpoint_path):
            logger.error(f"Best model checkpoint file not found at '{best_model_checkpoint_path}'. Cannot save final model.")
            return

        # --- 最终文件保存的目标路径 ---
        os.makedirs("./saved_models/", exist_ok=True)
        final_save_path = self._cfg.federate.save_to
        if not final_save_path:
            logger.error("`federate.save_to` is not specified in the config. Cannot save model.")
            return

        try:
            # --- 核心逻辑：获取最新 state_dict 并智能更新 ---
            logger.info(f"Reconstructing the full model for final saving (mode: {engine.aggregation_mode})...")

            # 1. 加载 EarlyStopper 保存的扁平化可训练参数向量
            best_trainable_params_vec_cpu = torch.load(best_model_checkpoint_path, map_location='cpu')

            # 2. 获取模型当前的 state_dict 作为最终要保存的模板和来源。
            #    这是最权威的键名和冻结参数来源。
            final_state_dict = self.model.state_dict()
            state_dict_keys = list(final_state_dict.keys())

            # 3. 恢复可训练参数
            #    a. 获取 engine 认为的、有序的可训练参数名
            trainable_param_names_from_engine = engine.ordered_trainable_names
            
            #    b. 动态查找这些参数在 state_dict 中对应的键
            matched_keys_in_state_dict_for_update = []
            for engine_name in trainable_param_names_from_engine:
                matched_key = None
                # 尝试多种匹配策略，以 endswith 为主，因为它对前缀不敏感
                for key in state_dict_keys:
                    if key == engine_name or engine_name.endswith("." + key) or key.endswith("." + engine_name):
                        # 检查更精确的后缀匹配，避免例如 "layer.weight" 错误匹配 "another_layer.weight"
                        if engine_name.split('.')[-1] == key.split('.')[-1]:
                            matched_key = key
                            break
                # Fallback to simple endswith if no precise match
                if not matched_key:
                    for key in state_dict_keys:
                        if engine_name.endswith(key):
                            matched_key = key
                            break
                
                if matched_key:
                    matched_keys_in_state_dict_for_update.append(matched_key)
                else:
                    raise KeyError(f"Could not find a matching key in state_dict for "
                                   f"engine parameter '{engine_name}'.")

            #    c. 创建一个临时的、与 state_dict 结构匹配的参数列表
            params_to_update = [final_state_dict[key] for key in matched_keys_in_state_dict_for_update]
            
            #    d. 调用 vector_to_parameters 将最佳参数写回
            #       这会就地更新 final_state_dict 中的张量
            vector_to_parameters(best_trainable_params_vec_cpu.to(params_to_update[0].device), params_to_update)

            # 4. 此时，final_state_dict 已经包含了所有冻结参数和更新后的可训练参数。
            #    我们可能需要对 key 进行归一化以匹配评估脚本的期望
            normalized_state_dict = OrderedDict()
            for key, value in final_state_dict.items():
                key_for_saving = key
                if key_for_saving.startswith('model.'):
                     key_for_saving = key_for_saving[len('model.'):]
                normalized_state_dict[key_for_saving] = value.cpu().clone()

            # 5. 按照 FS-LLM 的格式构建最终的检查点字典
            ckpt_to_save = {
                'cur_round': -1,
                'model': normalized_state_dict
            }
            
            # 6. 保存
            torch.save(ckpt_to_save, final_save_path)
            
            # 7. 清理
            os.remove(best_model_checkpoint_path)

            # --- 日志记录 ---
            best_score_info = ""
            if self.trainer.early_stopper.best_score is not None:
                best_score_info = f"Best eval loss: {self.trainer.early_stopper.best_score:.4f}. "
            
            logger.info(f"Client #{self.ID} (Rank 0): {best_score_info}Final full model (in state_dict format) saved to '{final_save_path}'.")

        except Exception as e:
            logger.error(f"Failed to save final compatible model to '{final_save_path}': {e}", exc_info=True)