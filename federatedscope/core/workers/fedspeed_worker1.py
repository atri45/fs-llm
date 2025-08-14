import logging
import os
import torch
import torch.distributed as dist

from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)

class FedSpeedServer(Server):
    """
    为 FedSpeed 模式定制的服务器，扮演 Coordinator 的角色。
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        # 将所有参数原封不动地传递给父类的构造函数
        super().__init__(
            ID=ID,
            state=state,
            config=config,
            data=data,
            model=model,
            client_num=client_num,
            total_round_num=total_round_num,
            device=device,
            strategy=strategy,
            unseen_clients_id=unseen_clients_id,
            **kwargs)

        # 用于 fedspeed_setup 建组的计数器
        self.fedspeed_ready_client_count = 0
        # 存储本轮参与方
        self.fedspeed_participants = []
        self.final_shards_buffer = {}

        self.register_handlers('fedspeed_ready',
                               self.callback_funcs_for_fedspeed_ready) 
        self.register_handlers('fedspeed_final_shard', self.callback_funcs_for_fedspeed_final_shard)

    def trigger_for_start(self):
        # 重写 trigger_for_start 方法
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
        world_size = len(participants)
        self.fedspeed_participants = sorted(list(self.comm_manager.neighbors.keys()))

        if world_size < 2: # FedSpeed 至少需要2个参与方
            logger.error(f"FedSpeed requires at least 2 clients, but only {world_size} joined.")
            self.terminate()
            return

        master_addr = self._cfg.distribute.server_host
        master_port = self._cfg.distribute.server_port + 100
        
        logger.info(f"FedSpeed Coordinator: master_addr={master_addr}, master_port={master_port}, world_size={world_size}")

        for i, client_id in enumerate(participants):
            rank = i
            dist_config = {
                'master_addr': master_addr,
                'master_port': master_port,
                'world_size': world_size,
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
        self.fedspeed_ready_client_count += 1
        logger.info(f"FedSpeed: Received 'ready' signal from Client #{message.sender}. "
                    f"({self.fedspeed_ready_client_count}/{len(self.comm_manager.neighbors)})")
        
        if self.fedspeed_ready_client_count == len(self.comm_manager.neighbors):
            logger.info("All clients are ready for FedSpeed! Starting parameter partitioning and distribution...")
            # 在这里触发下一步：参数分片与分发
            self.partition_and_distribute_model()

    # 我们将分片和分发合并到一个函数中，逻辑更清晰
    def partition_and_distribute_model(self):
        """
        计算分区地图，对模型进行分片，并将每个客户端专属的信息发送出去。
        """
        participants = self.fedspeed_participants
        num_clients = len(participants)

        # 1. 获取所有参数（包括冻结的和可训练的）的形状信息
        all_param_shapes = {name: param.shape for name, param in self.model.named_parameters()}
        trainable_param_names = {name for name, param in self.model.named_parameters() if param.requires_grad}

        # 2. 计算“分区地图” (Partition Map)
        #    这个地图告诉我们，每个参数被分成了几块，每块属于哪个rank
        partition_map = {}
        # shards_shapes[i] 将存储 rank i 应该持有的所有参数分片的形状
        shards_shapes_by_rank = [{} for _ in range(num_clients)]

        for name, full_shape in all_param_shapes.items():
            # 我们约定只对第0维进行切分
            dim_to_split = 0
            total_dim_size = full_shape[dim_to_split]
            
            # 如果参数太小无法切分，则由rank 0持有完整参数
            if total_dim_size < num_clients:
                partition_map[name] = [(0, full_shape)] # (rank, shape)
                shards_shapes_by_rank[0][name] = full_shape
                continue

            # 计算每个分片的精确大小
            split_sizes = [total_dim_size // num_clients] * num_clients
            for i in range(total_dim_size % num_clients):
                split_sizes[i] += 1
            
            partition_map[name] = []
            for i, shard_size in enumerate(split_sizes):
                shard_shape = list(full_shape)
                shard_shape[dim_to_split] = shard_size
                shard_shape = torch.Size(shard_shape)
                partition_map[name].append((i, shard_shape))
                shards_shapes_by_rank[i][name] = shard_shape
        
        # 3. 对模型参数进行物理切分
        shards_to_distribute = [{} for _ in range(num_clients)]
        for name, param in self.model.named_parameters():
            if name not in partition_map: continue

            # 获取该参数的分区信息
            partitions = partition_map[name]
            if len(partitions) == 1: # 不切分的参数
                rank = partitions[0][0]
                shards_to_distribute[rank][name] = param.data.cpu().clone()
                continue
            
            # 执行切分
            split_sizes = [shape[0] for _, shape in partitions]
            param_shards = list(torch.split(param.data.cpu(), split_sizes, dim=0))

            for i in range(len(partitions)):
                rank = partitions[i][0]
                shards_to_distribute[rank][name] = param_shards[i]

        # 4. 向每个客户端发送其专属的数据包
        for i, client_id in enumerate(participants):
            rank = i
            payload = {
                'partition_map': partition_map, # 完整的“分区地图”，每个客户端都需要
                'trainable_param_names': list(trainable_param_names), # 可训练参数列表
                'param_shard': shards_to_distribute[rank] # 只属于这个客户端的参数分片
            }
            
            # 使用 pickle + base64 来确保复杂数据结构的可靠传输
            import pickle, base64
            pickled_payload = pickle.dumps(payload)
            base64_payload = base64.b64encode(pickled_payload).decode('ascii')

            logger.info(f"Distributing partition info and param shard to Client #{client_id} (Rank {rank})")
            self.comm_manager.send(
                Message(msg_type='fedspeed_initial_shards',
                        sender=self.ID,
                        receiver=[client_id],
                        state=self.state,
                        content=base64_payload)
            )

    def callback_funcs_for_fedspeed_final_shard(self, message: Message):
        sender = message.sender
        final_shard_dict_from_msg = message.content
        import torch
        final_shard_dict = {
            name: torch.tensor(value) 
            for name, value in final_shard_dict_from_msg.items()
        }
        logger.info(f"Received final shard from Client #{sender}.")
        self.final_shards_buffer[sender] = final_shard_dict
        
        # 检查是否已收到所有分片
        if len(self.final_shards_buffer) == len(self.fedspeed_participants):
            logger.info("All final shards received. Reconstructing the final model...")
            
            # 重组模型
            # all_shards_list = [self.final_shards_buffer[client_id] for client_id in self.fedspeed_participants]
            # full_model_state_dict = self.reconstruct_from_shards(all_shards_list, self.device)
            # ... (这个重组逻辑需要细化，因为它需要知道rank和分片的对应关系)

            # 保存模型
            # final_model = get_model(self.cfg)
            # final_model.load_state_dict(full_model_state_dict, strict=False)
            # self.aggregator.save_model(...)
            
            logger.info("Final model reconstructed and saved. Terminating.")
            self.terminate()

class FedSpeedClient(Client):
    """
    为 FedSpeed 模式定制的客户端，作为分布式集群的一个 Worker。
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        # 将所有参数原封不动地传递给父类的构造函数
        super().__init__(
            ID=ID,
            server_id=server_id,
            state=state,
            config=config,
            data=data,
            model=model,
            device=device,
            strategy=strategy,
            is_unseen_client=is_unseen_client,
            *args,
            **kwargs)

        self.register_handlers('fedspeed_setup',
                               self.callback_funcs_for_fedspeed_setup)
        self.register_handlers('fedspeed_initial_shards', self.callback_funcs_for_initial_shards)

        self.my_rank = None
        self.partition_map = None
        self.dist_group_initialized = False

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

    def callback_funcs_for_initial_shards(self, message: Message):
        """
        接收来自服务器的初始化大礼包，包含分区地图和自己的参数分片。
        """
        import pickle, base64
        
        logger.info(f"Client #{self.ID} (Rank {self.my_rank}): Received initial shards and partition map.")
        
        # 1. 解码和反序列化
        base64_payload = message.content
        pickled_payload = base64.b64decode(base64_payload)
        payload = pickle.loads(pickled_payload)

        # 2. 存储分区地图和可训练参数名
        self.partition_map = payload['partition_map']
        param_shard = payload['param_shard']
        # 将可训练参数名传递给 Trainer
        self.trainer.trainable_param_names = payload['trainable_param_names']
        
        logger.info(f"  - Partition map received. This client holds {len(param_shard)} param shards.")

        # 3. 将本地模型变成一个只包含自己分片的“残缺”模型
        with torch.no_grad():
            # 遍历模型的所有参数
            for name, param in self.model.named_parameters():
                if name in param_shard:
                    # 如果这个参数的分片分配给了我
                    shard_data = param_shard[name]
                    # 创建一个正确形状的新参数，并替换掉旧的
                    # 这需要 param.data = ...，但直接替换Parameter对象更安全
                    new_param = torch.nn.Parameter(torch.empty(shard_data.shape, dtype=param.dtype, device=param.device))
                    setattr_by_name(self.model, name, new_param)
                else:
                    # 如果这个参数不归我管，就把它变成一个空的、0元素的张量
                    # 这样它就不占用内存，但在需要时仍然存在，避免AttributeError
                    # 注意：这是一种简化，更复杂的情况可能需要保留形状信息
                    empty_param = torch.nn.Parameter(torch.empty(0, dtype=param.dtype, device=param.device))
                    setattr_by_name(self.model, name, empty_param)
        
        # 4. 加载分片的权重
        self.model.load_state_dict(param_shard, strict=False)

        logger.info("Client's local model has been reshaped and loaded with its shards.")
        
        # 5. [验证] 打印一些分片信息
        for name, param in self.model.named_parameters():
             if param.numel() > 0: # 只打印非空的参数
                 logger.info(f"  - Final local shard '{name}' shape: {param.shape}")

        # 6. 一切准备就绪，可以开始训练了
        self.trainer.train()


# 一个辅助函数，用于通过字符串名称设置模块的属性
def setattr_by_name(obj, name, value):
    parts = name.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)