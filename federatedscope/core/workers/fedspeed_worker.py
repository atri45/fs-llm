import time
import torch
import psutil
import os
import logging

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



import logging
import os
import torch
import torch.distributed as dist
import pickle
import base64
from collections import OrderedDict

from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, add_prefix_to_path
from federatedscope.llm.trainer.fedspeed_engine import FedSpeedCoordinatorEngine
from federatedscope.llm.trainer.fedspeed_trainer import FedSpeedTrainer
from torch.nn.utils.convert_parameters import vector_to_parameters
from collections import OrderedDict
logger = logging.getLogger(__name__)

class FedSpeedServer(Server):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fedspeed_ready_client_count = 0
        self.world_size = 0
        self.fedspeed_participants = []
        self.coordinator_engine = None
        self.final_shards_buffer = {}

        # 我们现在只需要一个'ready'信号
        self.register_handlers('fedspeed_ready', self.callback_funcs_for_fedspeed_ready)
        self.register_handlers('fedspeed_final_shard', self.callback_funcs_for_fedspeed_final_shard)

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
        # 把我们之前设计的 fedspeed_setup 逻辑完整地搬到这里
        participants = list(self.comm_manager.neighbors.keys())
        self.world_size = len(participants)
        self.fedspeed_participants = sorted(list(self.comm_manager.neighbors.keys()))

        if self.world_size < 2: # FedSpeed 至少需要2个参与方
            logger.error(f"FedSpeed requires at least 2 clients, but only {self.world_size} joined.")
            self.terminate()
            return

        master_addr = self._cfg.distribute.server_host
        master_port = self._cfg.distribute.server_port + 100
        
        logger.info(f"FedSpeed Coordinator: master_addr={master_addr}, master_port={master_port}, world_size={self.world_size}")

        for i, client_id in enumerate(participants):
            rank = i
            dist_config = {
                'master_addr': master_addr,
                'master_port': master_port,
                'world_size': self.world_size,
                'rank': rank
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
        当所有客户端准备好后，在服务器端进行分片和演练，然后分发初始化包。
        """
        self.fedspeed_ready_client_count += 1
        logger.info(f"FedSpeed: Received 'ready' from Client #{message.sender}. ({self.fedspeed_ready_client_count}/{len(self.fedspeed_participants)})")
        
        if self.fedspeed_ready_client_count == len(self.fedspeed_participants):
            logger.info("All clients are ready! Server is now profiling and sharding the model...")
            
            # 1. 创建并运行协调引擎
            self.coordinator_engine = FedSpeedCoordinatorEngine(self.model, self._cfg, self.world_size)
            all_client_packages = self.coordinator_engine.profile_and_shard()
            self.coordinator_engine = None

            # 2. 向每个客户端发送其专属的初始化包
            for i, client_id in enumerate(self.fedspeed_participants):
                rank = i
                package_to_send = all_client_packages[rank]
                
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

    def callback_funcs_for_fedspeed_final_shard(self, message: Message):
        """
        收集所有客户端的最终分片，并在收集完毕后合成并保存完整模型。
        """
        sender_id = message.sender
        rank = self.fedspeed_participants.index(sender_id)
        
        logger.info(f"Received final shard from Client #{sender_id} (Rank {rank}).")
        
        # 1. 解码并存储分片
        content = pickle.loads(base64.b64decode(message.content))
        self.final_shards_buffer[rank] = content
        
        # 2. 检查是否所有分片都已收集完毕
        if len(self.final_shards_buffer) == len(self.fedspeed_participants):
            logger.info("All final shards received. Synthesizing the final model...")
            
            try:
                # 3. 合成模型
                # a. 按照 rank 顺序对收集到的分片进行排序
                sorted_shards_info = [self.final_shards_buffer[r] for r in range(self.world_size)]
                
                # b. 从中提取 FP32 分片并拼接成一个完整的、扁平化的向量
                fp32_shards = [info['fp32_master_shard'] for info in sorted_shards_info]
                full_fp32_vector = torch.cat(fp32_shards, dim=0)

                # c. 获取可训练参数的有序列表 (所有客户端的应该都一样，取第一个即可)
                ordered_trainable_names = sorted_shards_info[0]['ordered_trainable_names']

                # d. 准备一个干净的模型实例，用于加载最终权重
                final_model = self.model
                # 首先加载原始的、冻结的参数
                # (如果只微调adapter，这一步尤其重要，需要保留原始模型)
                # 我们的 `self.model` 在服务器上是完整的初始模型，正好可以用
                
                # e. 筛选出最终模型中可训练的参数
                trainable_params_in_final_model = [
                    p for name, p in final_model.named_parameters() 
                    if name in ordered_trainable_names
                ]

                # f. 【关键】将完整的 FP32 向量数据写回到模型的可训练参数中
                vector_to_parameters(full_fp32_vector, trainable_params_in_final_model)

                logger.info("Final model synthesized successfully.")

                # 4. 保存模型
                save_path_template = self._cfg.federate.save_to
                
                if not save_path_template:
                    logger.warning("`federate.save_to` is not configured. The final model will not be saved.")
                    # 如果未配置，则直接跳过保存
                else:
                    # 我们不加前缀，直接使用用户指定的文件名
                    # 如果用户想加前缀，应该在配置文件中指定
                    # 例如 save_to: "final_gpt2.ckpt"
                    final_model_path = save_path_template

                    # 检查路径中是否包含目录，如果包含，则创建
                    save_dir = os.path.dirname(final_model_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # 保存最终的、合成好的模型 state_dict
                    torch.save(final_model.state_dict(), final_model_path)
                    logger.info(f"Final synthesized model saved to {final_model_path}")

            except Exception as e:
                logger.error(f"Failed to synthesize or save the final model: {e}", exc_info=True)

            # 5. 结束联邦学习过程
            self.is_finish = True

class FedSpeedClient(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_rank = None
        self.dist_group_initialized = False
        self.monitor = PerformanceMonitor(self.device)

        self.register_handlers('fedspeed_setup', self.callback_funcs_for_fedspeed_setup)
        self.register_handlers('fedspeed_init_package', self.callback_funcs_for_init_package)
        
    def callback_funcs_for_fedspeed_setup(self, message: Message):
        if dist.is_initialized():
            logger.warning(f"Client #{self.ID}: Default process group already initialized. Destroying it before creating new FedSpeed group.")
            # 销毁已经存在的默认通信组
            dist.destroy_process_group()

        # 把我们之前设计的客户端建组逻辑完整地搬到这里
        if self.dist_group_initialized:
            logger.warning(f"Client #{self.ID} received fedspeed_setup, but already initialized.")
            return

        config = message.content
        self.my_rank = config['rank']

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
            logger.info(f"Client #{self.ID} (Rank {config['rank']}) successfully joined group. Waiting for initial shards...")
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
        【修改】接收初始化包，并启动基于 Step 的训练-评估循环。
        """
        logger.info(f"Client #{self.ID} (Rank {self.my_rank}): Received init package. Starting step-based run.")

        

        try:
            # 1. 初始化 Trainer 和 Engine
            init_package = pickle.loads(base64.b64decode(message.content))
            if not isinstance(self.trainer, FedSpeedTrainer):
                raise TypeError(f"FedSpeedClient is not configured with FedSpeedTrainer.")
            self.trainer.fedspeed_init_package = init_package

            # 3. 现在可以安全地开始训练了
            logger.info("All setup complete. Starting FedSpeed training...")
            self.monitor.start()
            self.trainer.train()

        finally:
            # 4. 无论训练是否成功，都尝试上传最终结果
            # 这确保了即使训练中途出错，我们也能保存当时的模型状态
            logger.info("Training finished or interrupted.")
            self.monitor.stop()
            self.upload_final_shard()
                
    def upload_final_shard(self):
        """
        从 FedSpeedEngine 中提取本地分片并上传。
        (此方法实现保持不变，是正确的)
        """
        logger.info(f"Client #{self.ID} uploading final parameter shard.")
        if not hasattr(self.trainer, 'fedspeed_engine') or self.trainer.fedspeed_engine is None:
            logger.error("Cannot upload shard, FedSpeedEngine not found.")
            return
            
        final_shard_dict = self.trainer.fedspeed_engine.get_my_final_shard()

        # 检查是否获取到了有效的分片
        if not final_shard_dict:
            logger.warning(f"Client #{self.ID} obtained an empty final shard dict. Nothing to upload.")
            return

        content = base64.b64encode(pickle.dumps(final_shard_dict)).decode('ascii')
        self.comm_manager.send(
            Message(msg_type='fedspeed_final_shard',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    content=content)
        )
        logger.info(f"Client #{self.ID} has sent its final shard.")

