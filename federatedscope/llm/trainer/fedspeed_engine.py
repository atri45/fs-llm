import threading
import queue
import torch
import torch.distributed as dist
import copy
import re
import os
import logging
import psutil
import numpy as np

from collections import OrderedDict
from functools import lru_cache
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)

class PipelinedReconstructor:
    def __init__(self, engine, name_to_param_map):
        self.engine = engine
        self.rank = engine.rank
        self.device = engine.device
        self.name_to_param_map = name_to_param_map
        self.param_name_to_exec_idx = {}
        self.direction = 'forward'

        # --- 配置 ---
        self.cache_path = engine.cfg.federate.get('model_cache_path')
        self.prefetch_depth = engine.cfg.federate.get('prefetch_depth', 2)
        self.gpu_mem_pressure_threshold = engine.cfg.federate.get('gpu_mem_pressure_threshold', 0.9)
        self.cpu_mem_pressure_threshold = engine.cfg.federate.get('cpu_mem_pressure_threshold', 0.9)

        if not self.cache_path:
            raise ValueError("`fedspeed.model_cache_path` must be specified.")

        # L1 缓存 (GPU)
        self.gpu_cache = OrderedDict()
        
        # L2 缓存 (CPU)
        self.cpu_cache = OrderedDict()

        # --- 预取流水线 ---
        self.prefetch_request_queue = queue.Queue(maxsize=self.prefetch_depth)
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._stop_prefetch = threading.Event()

    def load_param(self, param_name):
        """
        负责获取参数，并直接将其设置到模型参数 `param` 对象中，同时管理状态。
        """        
        # 1. 检查 L1 (GPU) 缓存
        if param_name in self.gpu_cache:
            logger.debug(f"Rank {self.engine.rank}: [GPU Cache Hit] for '{param_name}'.")
            self.gpu_cache.move_to_end(param_name)
            return self.gpu_cache[param_name]
            
        # 2. L1 未命中，检查 L2 (CPU) 缓存
        if param_name in self.cpu_cache:
            logger.debug(f"Rank {self.engine.rank}: [CPU Cache Hit] for '{param_name}'. Promoting to GPU Cache.")
            param_cpu = self.cpu_cache.pop(param_name)
            param_gpu = param_cpu.to(self.device, non_blocking=True)
            self._add_to_gpu_cache(param_name, param_gpu)
            return param_gpu
            
        # 3. L1 和 L2 都未命中，从磁盘加载
        key_to_find = param_name
        if key_to_find.startswith("base_model.model."):
            key_to_find = key_to_find[len("base_model.model."):]
        if key_to_find.startswith("model."):
            key_to_find = key_to_find[len("model."):]
        file_path = os.path.join(self.cache_path, key_to_find + ".npy")
        if not os.path.exists(file_path):
            logger.warning(f"Parameter file not found: {file_path}")
            return None
        
        numpy_array = np.load(file_path)
        param_cpu = torch.from_numpy(numpy_array)
        param_gpu = param_cpu.to(self.device, non_blocking=True)
        
        # 直接加入 L1 缓存
        self._add_to_gpu_cache(param_name, param_gpu)
        
        return param_gpu

    def _add_to_gpu_cache(self, param_name, param_tensor_gpu):
        """
        将一个 GPU 张量加入 L1 缓存，并将被淘汰的条目到 L2 (CPU) 缓存中。
        """
        param_size_bytes = param_tensor_gpu.numel() * param_tensor_gpu.element_size()
        
        # 循环淘汰，直到有足够空间
        while True:
            # 检查 VRAM 压力
            allocated = torch.cuda.memory_allocated(self.device)
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            
            if (allocated + param_size_bytes) / total_memory > self.gpu_mem_pressure_threshold:
                if not self.gpu_cache:
                    return # 无法缓存
                
                # 调用智能淘汰方法
                evicted_name, evicted_tensor = self._get_gpu_eviction_candidate()
                if evicted_name:
                    logger.debug(f"Rank {self.engine.rank}: [GPU Evict] '{evicted_name}' (dir: {self.direction}). Moving to L2.")
                    # 将被淘汰的条目存放到 L2 缓存
                    self._add_to_cpu_cache(evicted_name, evicted_tensor.cpu())
                    del evicted_tensor
            else:
                break
                
        self.gpu_cache[param_name] = param_tensor_gpu
        logger.debug(f"Rank {self.engine.rank}: [Disk Read] for '{param_name}'. Loading to GPU Cache.")
        if param_name in self.name_to_param_map:
            param_evicted = self.name_to_param_map[param_name]
            param_evicted.data = param_tensor_gpu
            param_evicted.fedspeed_status = "FULL"

    def _add_to_cpu_cache(self, param_name, param_tensor_cpu):
        """一个带动态淘汰逻辑的 L2 缓存添加方法。"""
        param_size_bytes = param_tensor_cpu.numel() * param_tensor_cpu.element_size()
        
        # 循环淘汰，直到有足够空间
        while True:
            # 检查当前 RAM 使用情况
            mem_info = psutil.virtual_memory()
            total_ram = mem_info.total
            used_ram = total_ram - mem_info.available
            
            # 如果预测的占用率超过了设定的阈值
            if (used_ram + param_size_bytes) / total_ram > self.cpu_mem_pressure_threshold:
                if not self.cpu_cache:
                    logger.debug(f"Cannot cache '{param_name}' to CPU. Not enough free RAM.")
                    return
                
                # 调用智能淘汰方法
                evicted_name, evicted_tensor = self._get_cpu_eviction_candidate()
                if evicted_name:
                    logger.debug(f"Rank {self.engine.rank}: [CPU Evict] '{evicted_name}' (dir: {self.direction}).")
                    del evicted_tensor
                if evicted_name in self.name_to_param_map:
                    param_evicted = self.name_to_param_map[evicted_name]
                    param_evicted.fedspeed_status = "ON_DISK"
                    logger.debug(f"evicted_name: {evicted_name}")
            else:
                # 空间足够
                break
                
        # 加入新条目
        self.cpu_cache[param_name] = param_tensor_cpu

    def _get_gpu_eviction_candidate(self):
        """【实现】为 L1 (GPU) 缓存决定要淘汰哪个条目。"""
        if not self.gpu_cache:
            return None, None

        victim_key = None
        if self.direction == 'forward':
            # 前向传播时，淘汰索引最小的 (最旧的)
            min_idx = float('inf')
            for name in self.gpu_cache.keys():
                idx = self.param_name_to_exec_idx.get(name, -1) # -1 表示非序列块参数
                if idx < min_idx:
                    min_idx = idx
                    victim_key = name
        else: # direction == 'backward'
            # 后向传播时，淘汰索引最大的 (最新的)
            max_idx = float('-inf')
            for name in self.gpu_cache.keys():
                idx = self.param_name_to_exec_idx.get(name, float('inf'))
                if idx > max_idx:
                    max_idx = idx
                    victim_key = name

        # 如果所有参数都没有索引 (例如模型很小，没有逻辑块)，则退化为 FIFO
        if victim_key is None:
            victim_key = next(iter(self.gpu_cache))

        return victim_key, self.gpu_cache.pop(victim_key)

    def _get_cpu_eviction_candidate(self):
        """【实现】为 L2 (CPU) 缓存决定要淘汰哪个条目。"""
        if not self.cpu_cache:
            return None, None

        victim_key = None
        if self.direction == 'forward':
            # 逻辑与 GPU 淘汰完全相同
            min_idx = float('inf')
            for name in self.cpu_cache.keys():
                idx = self.param_name_to_exec_idx.get(name, -1)
                if idx < min_idx:
                    min_idx = idx
                    victim_key = name
        else: # direction == 'backward'
            max_idx = float('-inf')
            for name in self.cpu_cache.keys():
                idx = self.param_name_to_exec_idx.get(name, float('inf'))
                if idx > max_idx:
                    max_idx = idx
                    victim_key = name
        
        if victim_key is None:
            victim_key = next(iter(self.cpu_cache))

        return victim_key, self.cpu_cache.pop(victim_key)

    def _prefetch_worker(self):
        """后台线程，负责从磁盘加载参数到预取队列。"""
        while not self._stop_prefetch.is_set():
            try:
                # 从请求队列中获取下一个要加载的参数名，最多等待1秒
                param_name_to_load = self.prefetch_request_queue.get(timeout=1)
                
                # 加载到缓存中
                self.load_param(param_name_to_load)
                    
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"Error in prefetch worker: {e}")

    def start(self):
        """启动预取线程。"""
        if self.prefetch_depth > 0:
            self._stop_prefetch.clear()
            self.prefetch_thread.start()
            logger.info(f"Rank {self.rank}: PipelinedReconstructor started with prefetch depth {self.prefetch_depth}.")

    def stop(self):
        """停止预取线程。"""
        if self.prefetch_depth > 0 and self.prefetch_thread.is_alive():
            self._stop_prefetch.set()
            # 清空请求队列以唤醒 worker
            while not self.prefetch_request_queue.empty():
                try: self.prefetch_request_queue.get_nowait()
                except queue.Empty: break
            self.prefetch_thread.join(timeout=2)
            logger.info(f"Rank {self.rank}: PipelinedReconstructor stopped.")

    def schedule_prefetch(self, param_names):
        """
        向后台线程提交一批预取请求。
        """
        if self.prefetch_depth > 0:
            for name in param_names:
                if self.prefetch_request_queue.full():
                    break # 请求队列满了，不再添加
                self.prefetch_request_queue.put(name)


class FedSpeedEngine:
    """
    【重构后】客户端引擎。
    它接收服务器预处理好的参数分片和引用计数，专注于执行训练循环。
    """
    def __init__(self, model, config, device, process_group, initial_shards, ref_counts):
        self.model = model
        self.cfg = config
        self.device = device
        self.comm_group = process_group or dist.group.WORLD
        self.rank = dist.get_rank(group=self.comm_group)
        self.world_size = dist.get_world_size(group=self.comm_group)
        self.current_step = 0
        self.reconstructor = None
        self._execution_order = []
        self._module_to_exec_idx = {}

        # 状态
        self.aggregation_mode = self.cfg.federate.get('aggregation_mode', 'local-reconstruct')
        self.aggregation_steps = self.cfg.federate.get('aggregation_steps', 1)
        self.local_cache_path = self.cfg.federate.get('model_cache_path')
        self.use_offline_model = self.cfg.federate.get('use_offline_model', False)
        self.two_stage = self.cfg.federate.get('two_stage_training', False)
        self.stage_one_steps = self.cfg.federate.get('stage_one_steps', 250)
        self.local_trainable_names_stage1 = None
        self.should_aggregate = False
        # for 'gradient' mode
        self.fp32_master_param_shard = None
        self.sharded_optimizer = None
        # for 'model'/'local-reconstruct' mode
        self.local_optimizer = None

        self._my_shard_info = {} # 用于切片
        self.temp_full_gradients = {}
        self.is_recomputing = False
        self.is_first_step = True
        self.gradients_mem = 0
        self.grad_clip_norm = 1.0
        
        # 引用计数 (从服务器接收)
        self.param_id_to_fwd_ref_count = ref_counts['fwd']
        self.param_id_to_recompute_ref_count = ref_counts['recompute']
        self.param_id_to_bwd_ref_count = ref_counts['bwd']
        self.param_fwd_ref_count = {}
        self.param_recompute_ref_count = {}
        self.param_bwd_ref_count = {}

        # 映射关系
        self.module_to_params_map = {}
        self.id_to_param_map = {}
        self.name_to_param_map = {}
        self.trainable_param_names = set()

        # --- 初始化引擎核心状态 ---
        self._attach_module_ids()
        if self.aggregation_mode == 'local-reconstruct':
            self._load_full_params(initial_shards.get('full_model_state_dict'))
            self.reconstructor = PipelinedReconstructor(self, self.name_to_param_map)
            self._build_sequential_execution_order()
        else:
            self._load_shards_and_build_maps(initial_shards, ref_counts['id_to_param_map_keys'])
        self.ordered_trainable_names = sorted(
            list(self.trainable_param_names),
            key=natural_sort_key
        )
        self._build_module_param_map()
        self._initialize_optimizer()
    
     # --- 初始化函数 ---

    def _load_shards_and_build_maps(self, initial_shards, id_to_param_map_keys):
        """用服务器发来的数据初始化参数状态和映射。"""
        # 建立 name -> param 映射
        temp_name_to_param = {name: p for name, p in self.model.named_parameters()}

        with torch.no_grad():
            for name, shard_info in initial_shards.items():
                param = temp_name_to_param[name]
                param.fedspeed_original_shape = shard_info['original_shape']
                if shard_info['is_sharded']:
                    # 这是一个分片参数 (冻结参数，或 gradient 模式下的所有参数)
                    param.fedspeed_status = "SHARDED"
                    param.fedspeed_shard = torch.nn.Parameter(
                        shard_info['shard_data'].to(self.device), 
                        requires_grad=False
                    )
                    param.data = torch.empty(0, dtype=param.fedspeed_shard.dtype, device=self.device)
                else:
                    # 这是一个完整的可训练参数 (只在 model 模式下出现)
                    param.fedspeed_status = "FULL_PERSISTENT"
                    # 将完整的参数数据直接加载到 param.data
                    param.data = shard_info['shard_data'].to(self.device)
                    # 这种参数没有 fedspeed_shard 属性
                    param.fedspeed_shard = None
                
                if shard_info['is_trainable']:
                    self.trainable_param_names.add(name)
                self.name_to_param_map[name] = param

        # 服务端和客户端的模型结构和参数ID必须一致
        # 用服务端发来的ID列表来重建id_to_param_map
        all_params = list(self.model.parameters())
        if len(id_to_param_map_keys) != len(all_params):
             raise ValueError("Mismatch in parameter count between server and client model.")
        for i, param_id in enumerate(id_to_param_map_keys):
             self.id_to_param_map[param_id] = all_params[i]

    def _load_full_params(self, state_dict):
        """
        根据模式，决定参数的初始状态 (SHARDED, ON_DISK, or FULL_PERSISTENT)。
        """
        if self.use_offline_model:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.fedspeed_name = name
                    param.fedspeed_original_shape = param.shape
                    is_trainable = param.requires_grad
                    if is_trainable:
                        param.fedspeed_status = "FULL_PERSISTENT"
                        param.fedspeed_shard = None
                        self.trainable_param_names.add(name)
                    else:
                        param.fedspeed_status = "ON_DISK"
                        param.data = torch.empty(0, dtype=param.dtype, device=self.device)
                        param.fedspeed_shard = None

                    self.name_to_param_map[name] = param
                    self.id_to_param_map[id(param)] = param
        else:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                # a. 归一化客户端本地的参数名，以生成查找 key
                    key_to_find = name
                    if key_to_find.startswith("base_model.model."):
                        key_to_find = key_to_find[len("base_model.model."):]
                    if key_to_find.startswith("model."):
                        key_to_find = key_to_find[len("model."):]

                    # b. 使用归一化后的 key 去模型的 state_dict 中查找
                    if key_to_find not in state_dict:
                        logger.warning(f"Parameter '{name}' (key: '{key_to_find}') not found in server state_dict. Skipping.")
                        continue
                    
                    # c. 找到了！现在可以安全地获取数据和元信息
                    server_param_data = state_dict[key_to_find]
                    param.fedspeed_name = name # 附加原始长名称
                    param.fedspeed_original_shape = server_param_data.shape
                    is_trainable = param.requires_grad
                
                    if is_trainable:
                        param.fedspeed_status = "FULL_PERSISTENT"
                        param.data = server_param_data.to(self.device)
                        param.fedspeed_shard = None
                        self.trainable_param_names.add(name)
                    else:
                        param.fedspeed_status = "ON_DISK"
                        param.data = torch.empty(0, dtype=param.dtype, device=self.device)
                        param.fedspeed_shard = None
                            
                    self.name_to_param_map[name] = param
                    self.id_to_param_map[id(param)] = param
        
            self._create_layer_cache_if_not_exists(state_dict)

    def _create_layer_cache_if_not_exists(self, state_dict):
        """在本地磁盘上创建分层缓存。"""
        if not self.local_cache_path:
            raise ValueError("`fedspeed.model_cache_path` must be specified for local-reconstruct mode.")
        
        done_file = os.path.join(self.local_cache_path, ".creation_done")
        if os.path.exists(done_file):
            logger.info(f"Rank {self.rank}: Layer-wise cache already exists at {self.local_cache_path}")
            return
            
        logger.info(f"Rank {self.rank}: Creating layer-wise cache at {self.local_cache_path}...")
        os.makedirs(self.local_cache_path, exist_ok=True)
        
        num_frozen_params_saved = 0
        for name, param in self.model.named_parameters():
            # 只处理冻结的参数
            if not param.requires_grad:
                # 归一化客户端的长名称，以匹配 state_dict 的 key
                key_in_state_dict = name
                if key_in_state_dict.startswith("base_model.model."):
                    key_in_state_dict = key_in_state_dict[len("base_model.model."):]
                if key_in_state_dict.startswith("model."):
                     key_in_state_dict = key_in_state_dict[len("model."):]
                
                # 确保这个冻结参数在服务器发来的 state_dict 中存在
                if key_in_state_dict in state_dict:
                    param_data_to_save = state_dict[key_in_state_dict]
                    param_filename = key_in_state_dict + ".npy"
                    full_path = os.path.join(self.local_cache_path, param_filename)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    # 将 PyTorch 张量转换为 NumPy 数组
                    numpy_array = param_data_to_save.cpu().numpy()
                    np.save(full_path, numpy_array)
                    num_frozen_params_saved += 1

        with open(done_file, 'w') as f: f.write('done')
        logger.info(f"Rank {self.rank}: Successfully created cache for {num_frozen_params_saved} frozen parameters.")

    def _attach_module_ids(self):
        """
        一个专门的函数，用于在所有操作之前，
        遍历一次模型并为每个模块贴上唯一的、持久化的 ID。
        """
        logger.debug(f"Rank {self.rank}: Attaching persistent fedspeed_id to all modules.")
        module_counter = 0
        for module in self.model.modules():
            module.fedspeed_id = module_counter
            module_counter += 1

    def _build_module_param_map(self):
        """
        这个函数现在只负责构建映射，它依赖于事先附加好的持久化 ID。
        """
        # 清空旧的映射，以备将来可能重新构建
        self.module_to_params_map.clear()
        
        for module in self.model.modules():
            # 不再检查或设置 ID，而是直接读取它
            # 确信这个属性在 __init__ 的第一步就已经被附加了
            module_id = module.fedspeed_id
            
            # 映射的 value 仍然是参数的内存地址 id()
            param_ids = [id(p) for p in module._parameters.values() if p is not None]
            
            # 使用持久化的 module_id 作为 key
            self.module_to_params_map[module_id] = param_ids

    def _build_sequential_execution_order(self):
        """
        遍历模型，找到所有的逻辑块 (例如 GPT2Block)，
        并构建一个有序的执行列表。
        """
        logger.info("Building sequential execution order for prefetching...")
        
        # 从配置中获取逻辑块的类名
        block_class_name = self.cfg.federate.get('transformer_block_class_name')
        if not block_class_name:
            raise ValueError("`fedspeed.transformer_block_class_name` must be specified.")
            
        # 动态地找到这个类
        block_class = None
        for module in self.model.modules():
            if module.__class__.__name__ == block_class_name:
                block_class = type(module)
                break
        
        if block_class is None:
            raise ValueError(f"Could not find module with class name '{block_class_name}' in the model.")
        
        logger.info(f"Identified transformer block class: {block_class.__name__}")

        # 遍历所有模块，只筛选出关心的逻辑块
        for module in self.model.modules():
            if isinstance(module, block_class):
                # 记录模块 ID 到其在执行顺序中索引的映射，用于快速查找
                self._module_to_exec_idx[module.fedspeed_id] = len(self._execution_order)
                # 将模块本身添加到有序列表中
                self._execution_order.append(module)
        
        # 构建一个 param_name -> exec_idx 的映射
        param_name_to_exec_idx = {}
        for idx, module in enumerate(self._execution_order):
            for p in module.parameters():
                if hasattr(p, 'fedspeed_name'):
                    # 将【块】的索引赋给它内部的【每个】参数
                    param_name_to_exec_idx[p.fedspeed_name] = idx
        self.reconstructor.param_name_to_exec_idx = param_name_to_exec_idx

        logger.info(f"Built execution order with {len(self._execution_order)} blocks.")

    def _initialize_optimizer(self):
        """
        【重构】根据 aggregation_mode 初始化相应的优化器和参数缓冲区。
        """        
        # --- 根据模式进行不同的初始化 ---
        if self.aggregation_mode == 'gradient':
            self._initialize_for_gradient_aggregation()
        elif self.aggregation_mode in ['model', 'local-reconstruct', 'fl-sim']:
            self._initialize_for_model_aggregation()
        else:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation_mode}")

    def _initialize_for_gradient_aggregation(self):
        """为 'gradient' 模式初始化分片优化器。"""
        logger.info("Initializing for 'gradient' aggregation mode with sharded optimizer.")
        trainable_shards = [self.name_to_param_map[name].fedspeed_shard for name in self.ordered_trainable_names]
        if not trainable_shards: return

        self.fp32_master_param_shard = parameters_to_vector(trainable_shards).clone().float().detach()
        self.fp32_master_param_shard.requires_grad = True
        
        # (复用之前写好的动态优化器创建逻辑)
        optimizer_config = self.cfg.train.optimizer
        optimizer_kwargs = copy.deepcopy(optimizer_config)
        optimizer_type = optimizer_kwargs.pop('type')
        lr = optimizer_kwargs.pop('lr')
        if '__help_info__' in optimizer_kwargs:
            del optimizer_kwargs['__help_info__']
        if '__cfg_check_funcs__' in optimizer_kwargs:
            del optimizer_kwargs['__cfg_check_funcs__']
        if 'is_ready_for_run' in optimizer_kwargs:
            del optimizer_kwargs['is_ready_for_run']
        if hasattr(torch.optim, optimizer_type):
            optimizer_class = getattr(torch.optim, optimizer_type)
        else:
            raise NotImplementedError(
                f"Optimizer '{optimizer_type}' is not implemented in torch.optim."
            )
        
        self.sharded_optimizer = optimizer_class([self.fp32_master_param_shard], lr=lr, **optimizer_kwargs)

    def _initialize_for_model_aggregation(self):
        """为 'model' 模式初始化完整优化器。"""
        logger.info("Initializing for 'model' aggregation mode with full local optimizer "
                    "acting on persistent full parameters.")
        
        # 1. 严格按照定义的顺序，从主模型中筛选出参数
        trainable_params_in_order = [
            self.name_to_param_map[name] 
            for name in self.ordered_trainable_names
        ]
        
        if not trainable_params_in_order:
            return

        # 2. 创建一个【完整】的本地优化器
        self.local_optimizer = get_optimizer(trainable_params_in_order, **self.cfg.train.optimizer)

        # 3. 如果开启了两阶段，则确定第一阶段要训练的参数【名称】
        if self.two_stage:
            self._partition_trainable_params_for_stage1()

    def _partition_trainable_params_for_stage1(self):
        """
        一个辅助函数，用于按逻辑块划分第一阶段的可训练参数。
        """
        logger.info(f"Rank {self.rank}: Partitioning trainable params for Stage 1.")
        
        # 1. 获取之前构建的、有序的逻辑块列表
        logical_blocks = self._execution_order
        if not logical_blocks:
            logger.warning("No logical blocks found in execution order. "
                         "Stage 1 will train ALL parameters on all clients.")
            # 如果找不到逻辑块，则退化为所有客户端都训练所有参数
            self.local_trainable_names_stage1 = set(self.ordered_trainable_names)
            return

        num_blocks = len(logical_blocks)
        
        # 2. 对【逻辑块列表】进行均匀分块
        #    例如，12 个块分给 3 个客户端
        block_indices = torch.arange(num_blocks)
        block_chunks = torch.chunk(block_indices, self.world_size)
        my_block_indices = block_chunks[self.rank]
        
        # 3. 获取分配给当前 rank 的逻辑块实例
        my_blocks = [logical_blocks[i] for i in my_block_indices]
        
        # 4. 从这些块中，提取出所有可训练参数的【全局名称】
        my_param_names = set()
        for block in my_blocks:
            for name, param in block.named_parameters(recurse=True):
                if param.requires_grad:
                    # 需要参数的全局名称，而不是它在块内部的局部名称
                    # 之前已经为每个 param 附加了 .fedspeed_name
                    if hasattr(param, 'fedspeed_name'):
                        my_param_names.add(param.fedspeed_name)
        
        self.local_trainable_names_stage1 = my_param_names
        
        # 打印分配结果以供验证
        logger.info(f"Rank {self.rank}: In Stage 1, assigned {len(my_blocks)} logical blocks "
                    f"(Indices: {my_block_indices.tolist()}), "
                    f"resulting in {len(self.local_trainable_names_stage1)} trainable parameters.")

    def _register_hooks_recursively(self, module: torch.nn.Module, prefix=''):
        """递归注册钩子。"""
        if self.aggregation_mode != 'fl-sim':
            module.register_forward_pre_hook(self._pre_forward_hook_fn)
            module.register_forward_hook(self._post_forward_hook_fn)
            module.register_full_backward_pre_hook(self._pre_backward_hook_fn)
            module.register_full_backward_hook(self._post_backward_hook_fn)
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                fullname = prefix + name
                param.register_hook(self._collect_grad_hook_factory(fullname))
        for name, child in module.named_children():
            self._register_hooks_recursively(child, prefix=prefix + name + '.')

    # --- 核心钩子函数 ---

    def _pre_forward_hook_fn(self, module, inputs):
        """
        根据参数状态，分发到不同的重建方法。
        """
        logger.debug(f"[PRE-FWD HOOK @ Rank {self.rank}] For {type(module).__name__}")
        with torch.no_grad():
            # 遍历模块的【所有】参数，检查它们的状态
            for name, param in module.named_parameters(recurse=False):
                if param is not None and hasattr(param, 'fedspeed_status'):
                    if param.fedspeed_status == "SHARDED":
                        self._network_reconstruct_param(param)
                    elif param.fedspeed_status == "ON_DISK":
                        self._local_reconstruct_param(param)
                    # 如果状态是 FULL_PERSISTENT 或 FULL，则什么都不做

    def _local_reconstruct_param(self, param):
        """从本地分层缓存加载冻结参数。"""
        if not self.reconstructor:
            logger.error("LocalReconstructor not initialized. Cannot perform local reconstruction.")
            return
        
        param_name = param.fedspeed_name
        param.data = self.reconstructor.load_param(param_name)

    def _network_reconstruct_param(self, param):
        """通过网络通信 (all-gather)，动态地将一个【分片】的参数重建为【完整】形态。"""
        # 如果这个参数在所有 rank 上都是空的，则跳过
        if param.fedspeed_original_shape[0] == 0:
            return
        
        # 如果参数太小，只在 rank 0 上，那么需要广播
        if param.fedspeed_original_shape[0] < self.world_size:
            full_param_tensor = torch.empty(param.fedspeed_original_shape, 
                                            dtype=param.fedspeed_shard.dtype, 
                                            device=self.device)
            # rank 0 持有数据，将其广播给其他人
            src_rank = 0
            dist.broadcast(full_param_tensor, src=src_rank, group=self.comm_group)
            param.data = full_param_tensor
            param.fedspeed_status = "FULL"
            return
            
        # 将本地分片扁平化
        local_shard_flat = param.fedspeed_shard.data.flatten()
        
        # 1. 【第一步：尺寸同步】
        # 获取本地扁平化分片的元素数量
        my_numel = local_shard_flat.numel()
        # 准备一个 tensor 来接收所有 rank 的 numel
        all_numels = [torch.tensor(0, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
        # 使用 all_gather 收集所有人的 numel
        dist.all_gather(all_numels, torch.tensor(my_numel, dtype=torch.long, device=self.device))
        all_numels_list = [t.item() for t in all_numels]

        # 2. 【第二步：数据收集】
        # 2a. 计算所有分片的最大尺寸
        max_numel = max(all_numels_list)

        # 2b. 填充本地分片到最大尺寸
        padded_local_shard = torch.zeros(max_numel, dtype=local_shard_flat.dtype, device=self.device)
        padded_local_shard[:my_numel] = local_shard_flat
        
        # 2c. 准备输出张量，它将包含所有填充后的分片
        # 每个 rank 都会得到一个一模一样的、包含所有数据的张量
        all_padded_shards_flat = torch.empty(max_numel * self.world_size, dtype=local_shard_flat.dtype, device=self.device)

        # 2d. 执行 all_gather
        # 输入张量 (padded_local_shard) 在所有 rank 上形状相同 (max_numel,)
        # 输出张量 (all_padded_shards_flat) 也在所有 rank 上形状相同
        dist.all_gather_into_tensor(all_padded_shards_flat, padded_local_shard)
        
        # 3. 【第三步：手动切片和重组】
        # 从巨大的扁平张量中，根据真实的尺寸信息 (all_numels_list)，切出每个 rank 的有效数据，并拼接起来。
        tensors_to_cat = []
        current_offset = 0
        for i in range(self.world_size):
            # rank i 的真实数据长度
            real_numel = all_numels_list[i]
            # 从填充后的大张量中，切出 rank i 的填充后分片
            padded_shard_i = all_padded_shards_flat.narrow(0, i * max_numel, max_numel)
            # 从中只取有效部分
            real_shard_i = padded_shard_i.narrow(0, 0, real_numel)
            tensors_to_cat.append(real_shard_i)
        
        # 拼接成完整的、扁平化的参数
        full_param_flat = torch.cat(tensors_to_cat, dim=0)

        # 恢复原始形状
        param.data = full_param_flat.view(param.fedspeed_original_shape)
        param.fedspeed_status = "FULL"

    def _post_forward_hook_fn(self, module, inputs, outputs):
        """在前向传播离开一个子模块之后，立即丢弃不再使用的参数。"""
        logger.debug(f"[POST-FWD HOOK @ Rank {self.rank}] For {type(module).__name__}")

        # 1. 参数释放
        # if self.aggregation_mode == 'model':
        #     self._release_all_full_params()
        if self.aggregation_mode == 'gradient':
            param_ids_in_module = self.module_to_params_map.get(id(module), [])
            current_ref_counter = self.param_recompute_ref_count if self.is_recomputing else self.param_fwd_ref_count
            with torch.no_grad():
                for param_id in param_ids_in_module:              
                    if param_id in current_ref_counter:
                        current_ref_counter[param_id] -= 1
                        if current_ref_counter[param_id] == 0:
                            param = self.id_to_param_map[param_id]
                            if hasattr(param, 'fedspeed_status') and param.fedspeed_status == "FULL":
                                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
                                param.fedspeed_status = "SHARDED"
                                del current_ref_counter[param_id]

        # 2. 参数预读取
        if self.aggregation_mode == 'local-reconstruct':
            self.reconstructor.direction = 'forward'
            # a. 获取接下来要预取的模块 (方向是 forward)
            next_modules = self._get_modules_for_prefetch(module, is_forward=True)
            # b. 调度预取
            self._schedule_prefetch_for_modules(next_modules)

    def _get_modules_for_prefetch(self, current_module, is_forward):
        """
        【重构】根据当前模块和传播方向，获取接下来需要预取的模块。
        """
        current_idx = self._module_to_exec_idx.get(current_module.fedspeed_id)
        if current_idx is None: return []
            
        prefetch_depth = self.cfg.federate.get('prefetch_depth', 1)
        
        if is_forward:
            # 前向传播：预取【之后】的模块
            start_idx = current_idx + 1
            end_idx = start_idx + prefetch_depth
            return self._execution_order[start_idx:end_idx]
        else: # is_backward
            # 后向传播：预取【之前】的模块
            start_idx = current_idx - prefetch_depth
            end_idx = current_idx
            # 确保索引不越界
            start_idx = max(0, start_idx)
            # 返回的是逆序的切片，但顺序不重要，因为只关心集合
            return self._execution_order[start_idx:end_idx]

    def _schedule_prefetch_for_modules(self, modules_to_prefetch):
        """
        一个辅助函数，用于为一批模块调度预取。
        """
        if not modules_to_prefetch or not self.reconstructor: return
        
        params_to_prefetch = []
        for module in modules_to_prefetch:
            for name, p in module.named_parameters(recurse=True):
                if p is not None and hasattr(p, 'fedspeed_status') and p.fedspeed_status == "ON_DISK":
                    params_to_prefetch.append(p.fedspeed_name)
        
        if params_to_prefetch:
            unique_params = sorted(list(set(params_to_prefetch)))
            logger.debug(f"Rank {self.rank}: Scheduling prefetch for {len(unique_params)} params.")
            self.reconstructor.schedule_prefetch(unique_params)

    def _pre_backward_hook_fn(self, module, grad_output):
        """在反向传播进入一个子模块之前，重建该模块所需的所有参数。"""
        logger.debug(f"[PRE-BWD HOOK @ Rank {self.rank}] For {type(module).__name__}")

        # 复用 pre_forward 的参数重建逻辑
        self._pre_forward_hook_fn(module, None)

    def _post_backward_hook_fn(self, module, grad_input, grad_output):
        """在反向传播离开一个子模块之后，立即丢弃不再使用的参数。"""
        logger.debug(f"[POST-BWD HOOK @ Rank {self.rank}] For {type(module).__name__}")

        # 1. 参数释放
        # if self.aggregation_mode == 'model':
        #     self._release_all_full_params()
        if self.aggregation_mode == 'gradient':
            param_ids_in_module = self.module_to_params_map.get(id(module), [])
            current_ref_counter = self.param_recompute_ref_count if self.is_recomputing else self.param_bwd_ref_count
            with torch.no_grad():
                for param_id in param_ids_in_module:              
                    if param_id in current_ref_counter:
                        current_ref_counter[param_id] -= 1
                        if current_ref_counter[param_id] == 0:
                            param = self.id_to_param_map[param_id]
                            if hasattr(param, 'fedspeed_status') and param.fedspeed_status == "FULL":
                                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
                                param.fedspeed_status = "SHARDED"
                                del current_ref_counter[param_id]

        # 2. 参数预读取
        if self.aggregation_mode == 'local-reconstruct':
            self.reconstructor.direction = 'backward'
            # a. 获取接下来要预取的模块 (方向是 forward)
            next_modules = self._get_modules_for_prefetch(module, is_forward=False)
            # b. 调度预取
            self._schedule_prefetch_for_modules(next_modules)

    def _collect_grad_hook_factory(self, name):
        def hook(grad):
            if grad is not None:
                self.temp_full_gradients[name] = grad.detach().clone()
        return hook
    
    # --- 核心训练方法 ---

    def forward(self, *args, **kwargs):    
        logger.debug(f"Rank {self.rank}: Starting forward pass...")

        # 加载预先计算好的计数
        self.param_fwd_ref_count = self.param_id_to_fwd_ref_count.copy()

        # 前向传播
        outputs = self.model(*args, **kwargs)

        # 释放参数
        if self.aggregation_mode == 'gradient':
            self._release_all_full_params()
        return outputs

    def backward(self, loss, *args, **kwargs):
        logger.debug(f"Rank {self.rank}: Starting backward pass...")

        # 加载预先计算好的计数
        self.param_bwd_ref_count = self.param_id_to_bwd_ref_count.copy()
        self.param_recompute_ref_count = self.param_id_to_recompute_ref_count.copy()

        # 反向传播
        loss.backward()

        # 确保所有客户端的backward和所有钩子都已执行完毕
        # dist.barrier()

        # 计算梯度大小
        if self.is_first_step:
            for grad in self.temp_full_gradients.values():
                self.gradients_mem += grad.numel() * grad.element_size()
                
        # 释放参数
        if self.aggregation_mode == 'gradient':
            self._release_all_full_params()

        logger.debug(f"Rank {self.rank}: Backward pass and all gradient hooks finished.")

    def aggregate_gradients(self):
        """
        使用 all_reduce + 本地切片的方式，兼容 gloo 后端。
        """
        if self.aggregation_mode in ['model', 'local-reconstruct', 'fl-sim']:
            # 在模型聚合模式下，梯度只在本地使用，不进行分布式聚合
            return None 
        else:
            ordered_full_grads = [self.temp_full_gradients[name] for name in self.ordered_trainable_names]
            
            # 1. 打包：将所有完整梯度拉平成一个巨大的、连续的向量
            flat_full_grads = parameters_to_vector(ordered_full_grads)

            # 2.通信：使用 All-Reduce
            dist.all_reduce(flat_full_grads, op=dist.ReduceOp.SUM)
            if self.world_size > 0:
                flat_full_grads.div_(self.world_size)
            
            # 3. 本地切片 
            # 3a. 获取所有 rank 的分片大小 (numel)，以确定自己的偏移量
            my_shard_numel = self.fp32_master_param_shard.numel()
            all_shard_numels_t = [torch.tensor(0, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(all_shard_numels_t, torch.tensor(my_shard_numel, dtype=torch.long, device=self.device))
            split_sizes = [t.item() for t in all_shard_numels_t]

            # 3b. 计算本地分片在完整向量中的偏移量
            #    偏移量等于所有排在自己前面的 rank 的分片大小之和
            my_offset = sum(split_sizes[:self.rank])
            
            # 3c. 从完整的聚合梯度中，切出属于自己的分片
            flat_grad_shard = flat_full_grads.narrow(0, my_offset, my_shard_numel)
            
            # 清理工作
            self.temp_full_gradients.clear()
            
            # 返回切片后的梯度。注意，它现在是 flat_full_grads 的一个视图 (view)。
            # 为了安全，返回一个克隆，以防 flat_full_grads 被意外修改。
            return flat_grad_shard.clone()

    def step(self, flat_grad_shard: torch.Tensor = None):
        """
        【重构】根据 self.aggregation_mode 选择不同的更新路径。
        注意：当模式为 'model' 时，传入的 flat_grad_shard 将被忽略。
        """
        if self.aggregation_mode in ['model', 'local-reconstruct']:
            return self._local_reconstruct_step()
        elif self.aggregation_mode == 'gradient':
            self._gradient_aggregation_step(flat_grad_shard)
        elif self.aggregation_mode == 'fl-sim':
            return self._fl_sim_step()
        else:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation_mode}")

    def _local_reconstruct_step(self):
        """
        'local-reconstruct' 模式的 step 逻辑，支持同步和异步聚合。
        """
        if not self.local_optimizer: return None

        # --- 阶段性梯度处理 ---
        if self.two_stage and self.current_step < self.stage_one_steps:
            # --- 第一阶段：梯度掩码 ---
            with torch.no_grad():
                # 遍历优化器管理的所有参数
                for param in self.local_optimizer.param_groups[0]['params']:
                    # 找到参数的全局名称
                    param_name = param.fedspeed_name # 假设 param 对象有这个属性
                    
                    # 如果这个参数不属于本地子集，则将其梯度清零
                    if param_name not in self.local_trainable_names_stage1:
                        if param.grad is not None:
                            param.grad.zero_()

        # 1. 本地更新
        # 梯度已经在 backward 过程中被附加到了【完整】可训练参数上，直接调用优化器执行一步本地更新
        self.local_optimizer.step()
        self.local_optimizer.zero_grad()
        self.current_step += 1

        # 打印内存占用
        if self.current_step == 1:
            self.profile_sharded_memory()

        # # 2. 检查是否需要聚合
        self.should_aggregate = False
        if self.two_stage:
            if self.current_step == self.stage_one_steps:
                # 阶段转换点，强制聚合
                self.should_aggregate = True
                logger.info(f"--- [Stage Transition] Step #{self.current_step} ---")
            elif self.current_step > self.stage_one_steps and self.current_step % self.aggregation_steps == 0:
                # 第二阶段，按频率聚合
                self.should_aggregate = True
            # else: 第一阶段，不聚合
        else: # 非两阶段模式
            if self.current_step % self.aggregation_steps == 0:
                self.should_aggregate = True
        
        # 返回要发送的参数
        if self.should_aggregate:
            trainable_params = self.local_optimizer.param_groups[0]['params']
            params_to_send_vec = parameters_to_vector([p.data for p in trainable_params]).clone()

            # 清理未使用的缓存内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return params_to_send_vec
        
        # 3. 清理未使用的缓存内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None

    def _gradient_aggregation_step(self, flat_grad_shard: torch.Tensor):
        """执行优化器步骤。用聚合后的梯度分片来更新FP32主权重分片。"""
        if not self.sharded_optimizer or flat_grad_shard.numel() == 0:
            logger.warning("Optimizer not ready or no gradients, skipping step.")
            return
        logger.debug(f"Rank {self.rank}: Starting optimizer step...")

        # 1. 将这个扁平化的梯度分片，赋给FP32主权重分片的.grad属性。
        self.fp32_master_param_shard.grad = flat_grad_shard.to(self.fp32_master_param_shard.device)

        # 梯度裁剪
        # if self.grad_clip_norm > 0:
        #     clip_grad_norm_(
        #         self.fp32_master_param_shard, 
        #         max_norm=self.grad_clip_norm
        #     )

        # 2. 调用分片化的优化器，执行一步更新！
        self.sharded_optimizer.step()
        self.sharded_optimizer.zero_grad() 
        self.current_step += 1

        # 打印内存占用
        if self.current_step == 1:
            self.profile_sharded_memory()

        # 重置优化器    
        # self.sharded_optimizer.state.clear()
        
        # 3. 将更新后的FP32主权重，写回到模型参数的FP16分片中。
        shards_to_update = [self.name_to_param_map[name].fedspeed_shard for name in self.ordered_trainable_names]
        vector_to_parameters(self.fp32_master_param_shard.data, shards_to_update)

        # 4. 清理未使用的缓存内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug(f"Rank {self.rank}: Optimizer step finished. Model shards updated.")
        
    def _model_aggregation_step(self):
        """
        【可训练参数不分片版】根据步数，条件性地执行聚合。
        """
        if not self.local_optimizer: return

        # --- 阶段性梯度处理 ---
        if self.two_stage and self.current_step < self.stage_one_steps:
            # --- 第一阶段：梯度掩码 ---
            with torch.no_grad():
                # 遍历优化器管理的所有参数
                for param in self.local_optimizer.param_groups[0]['params']:
                    # 找到参数的全局名称
                    param_name = param.fedspeed_name # 假设 param 对象有这个属性
                    
                    # 如果这个参数不属于本地子集，则将其梯度清零
                    if param_name not in self.local_trainable_names_stage1:
                        if param.grad is not None:
                            param.grad.zero_()

        # 1. 本地更新
        # 梯度已经在 backward 过程中被附加到了【完整】可训练参数上，直接调用优化器执行一步本地更新
        self.local_optimizer.step()
        self.local_optimizer.zero_grad()
        self.current_step += 1
        
        # 打印内存占用
        if self.current_step == 1:
            self.profile_sharded_memory()

        # 2. 检查是否需要聚合
        self.should_aggregate = False
        if self.two_stage:
            if self.current_step == self.stage_one_steps:
                # 阶段转换点，强制聚合
                self.should_aggregate = True
                logger.info(f"--- [Stage Transition] Aggregating at Step #{self.current_step} ---")
            elif self.current_step > self.stage_one_steps and self.current_step % self.aggregation_steps == 0:
                # 第二阶段，按频率聚合
                self.should_aggregate = True
                logger.info(f"--- [Stage 2] Aggregating at Step #{self.current_step} ---")
            # else: 第一阶段，不聚合
        else: # 非两阶段模式
             if self.current_step % self.aggregation_steps == 0:
                 self.should_aggregate = True
                 logger.info(f"--- [FedSpeed-Model] Aggregating at Step #{self.current_step} ---")

        if self.should_aggregate:            
            # a. 将本地更新后的【完整】可训练参数打包成一个向量
            trainable_params_in_order = self.local_optimizer.param_groups[0]['params']
            updated_local_full_params_vec = parameters_to_vector(
                [p.data for p in trainable_params_in_order]
            )

            # b. 对该向量进行 all-reduce 平均 (SUM + DIV)
            dist.all_reduce(updated_local_full_params_vec, op=dist.ReduceOp.SUM)
            if self.world_size > 0:
                updated_local_full_params_vec.div_(self.world_size)

            # c. 将聚合后的参数写回到模型中
            vector_to_parameters(updated_local_full_params_vec, trainable_params_in_order)

            # 聚合后重置优化器
            # self._initialize_for_model_aggregation()

        # 3. 清理未使用的缓存内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug(f"Rank {self.rank}: Optimizer step finished. Model shards updated.")

    def _fl_sim_step(self):
        """
        完全模拟传统 FL 的 step 逻辑。
        """
        if not self.local_optimizer: return None

        # 1. 本地更新 (在高精度上进行)
        self.local_optimizer.step()
        self.local_optimizer.zero_grad()
        self.current_step += 1
        
        # 2. 打印内存占用
        if self.current_step == 1:
            self.profile_sharded_memory()

        # 3. 检查是否需要聚合
        self.should_aggregate = False
        if self.current_step % self.aggregation_steps == 0:
            self.should_aggregate = True
            trainable_params = self.local_optimizer.param_groups[0]['params']
            params_to_send_vec = parameters_to_vector([p.data for p in trainable_params]).clone()

            # 清理未使用的缓存内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return params_to_send_vec

        # 4. 清理未使用的缓存内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return None

    def update_model(self, received_params_vec: torch.Tensor):
        """由 Trainer 调用的、用于【同步】更新的方法。"""
        with torch.no_grad():
            trainable_params_in_order = [self.name_to_param_map[name] for name in self.ordered_trainable_names]
            vector_to_parameters(received_params_vec.to(self.device), trainable_params_in_order)

    def aggregate_model(self, received_params_vec: torch.Tensor, alpha: float):
        """
        由 Trainer 调用的、纯计算的聚合方法。
        """
        with torch.no_grad():
            trainable_params_in_order = [self.name_to_param_map[name] for name in self.ordered_trainable_names]
            current_params_vec = parameters_to_vector([p.data for p in trainable_params_in_order])
            current_params_vec.mul_(1 - alpha).add_(received_params_vec.to(self.device), alpha=alpha)
            vector_to_parameters(current_params_vec, trainable_params_in_order)

    def get_full_trainable_params_as_vector(self) -> torch.Tensor:
        """
        直接从【本地优化器】中获取完整的、高精度的可训练参数，并返回一个扁平化的向量。
        """
        if not self.local_optimizer:
            logger.warning("`get_full_trainable_params_as_vector` called but local_optimizer is not initialized.")
            return torch.tensor([])

        # 优化器管理的参数就是最新的 FP32 状态
        params_to_vectorize = self.local_optimizer.param_groups[0]['params']
        return parameters_to_vector([p.data for p in params_to_vectorize]).clone()

    def set_full_trainable_params_from_vector(self, full_vector: torch.Tensor):
        """
        从一个完整的 FP32 向量中，恢复所有可训练参数的状态。
        """
        if not self.local_optimizer:
            logger.warning("`set_full_trainable_params_from_vector` called but local_optimizer is not initialized.")
            return
        
        # 1. 直接将数据写入由优化器管理的参数
        params_to_update = self.local_optimizer.param_groups[0]['params']
        vector_to_parameters(full_vector, params_to_update)

        # 2. 将更新后的高精度参数，写回到【低精度】的模型中
        with torch.no_grad():
            trainable_params_in_model = [
                self.name_to_param_map[name] 
                for name in self.ordered_trainable_names
            ]
            
            params_to_write_vec = parameters_to_vector(params_to_update)
            params_to_write_vec = params_to_write_vec.to(trainable_params_in_model[0].dtype)
            vector_to_parameters(params_to_write_vec, trainable_params_in_model)
        
    def profile_sharded_memory(self):
        """计算各类分片化状态的内存占用。"""
        if not torch.cuda.is_available():
            return
            
        # 1. 模型参数分片 (FP16/BF16)
        param_shards_mem = 0
        for param in self.model.parameters():
            if hasattr(param, 'fedspeed_shard'):
                # 检查参数是分片的还是完整的
                if hasattr(param, 'fedspeed_status') and param.fedspeed_status == "SHARDED":
                    param_shards_mem += param.fedspeed_shard.numel() * param.fedspeed_shard.element_size()
                elif hasattr(param, 'fedspeed_status') and param.fedspeed_status == "FULL_PERSISTENT":
                    # 这是 'model' 模式下的完整可训练参数
                    param_shards_mem += param.numel() * param.element_size()

        # 2. 优化器状态分片
        optimizer_states_mem = 0
        if self.aggregation_mode == 'gradient':
            if self.sharded_optimizer:
                for state in self.sharded_optimizer.state.values():
                    for s in state.values():
                        if isinstance(s, torch.Tensor):
                            optimizer_states_mem += s.numel() * s.element_size()
        elif self.aggregation_mode in ['model', 'local-reconstruct', 'fl-sim']:
            if self.local_optimizer:
                for state in self.local_optimizer.state.values():
                    for s in state.values():
                        if isinstance(s, torch.Tensor):
                            optimizer_states_mem += s.numel() * s.element_size()

        logger.info("--- [Memory Profile (Sharded States)] ---")
        logger.info(f"  - Model Param Shards (FP16/BF16): {param_shards_mem / (1024**2):.2f} MB")
        logger.info(f"  - Full Gradients: {self.gradients_mem / (1024**2):.2f} MB")
        logger.info(f"  - Optimizer States Shards (Adam):  {optimizer_states_mem / (1024**2):.2f} MB")

    def _release_all_full_params(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if hasattr(param, 'fedspeed_status') and param.fedspeed_status == "FULL":
                    # 1. 释放 GPU 显存
                    param.data = torch.empty(0, dtype=param.dtype, device=self.device)
                    # 2. 根据当前的聚合模式，将其恢复到正确的初始状态
                    if self.aggregation_mode == 'local-reconstruct':
                        # 在本地重建模式下，被重建的都是冻结参数，它们来自磁盘
                        param.fedspeed_status = "ON_DISK"
                    else:
                        # 在 'gradient' 或 'model' (ZeRO) 模式下，被重建的参数来自网络分片
                        param.fedspeed_status = "SHARDED"


class FedSpeedCoordinatorEngine:
    """
    服务器端协调引擎。
    负责在训练开始前，对完整模型进行所有预处理：
    1. 参数分片。
    2. 包装激活检查点。
    3. 执行一次完整的前向和后向传播“演练”，以计算所有参数的引用计数。
    4. 将每个客户端所需的分片和引用计数打包。
    """
    def __init__(self, model, config, world_size, device='cpu'):
        self.model = model.to(device) # 演练可以在CPU上进行，节省GPU
        self.cfg = config
        self.world_size = world_size
        self.device = device

        # 演练所需的状态
        self.param_id_to_fwd_ref_count = {}
        self.param_id_to_recompute_ref_count = {}
        self.param_id_to_bwd_ref_count = {}
        self.is_recomputing = False
        
        # 模块和参数映射
        self.module_to_params_map = {}
        self.id_to_param_map = {}
        self._build_module_param_map()

    def _build_module_param_map(self):
        """在初始化时，遍历一次模型，构建模块到其直属参数的映射。"""
        for module in self.model.modules():
            if not hasattr(module, 'fedspeed_id'):
                module.fedspeed_id = id(module)
            params_in_module = [id(p) for p in module._parameters.values() if p is not None]
            self.module_to_params_map[module.fedspeed_id] = params_in_module
            for param in module._parameters.values():
                if param is not None:
                    self.id_to_param_map[id(param)] = param

    def _apply_checkpointing_for_profile(self, block_class):
        """为演练模型应用激活检查点。"""
        logger.info(f"[Coordinator] Applying activation checkpointing to {block_class.__name__} for profiling.")
        from torch.utils.checkpoint import checkpoint
        for module_to_patch in self.model.modules():
            if isinstance(module_to_patch, block_class):
                module_to_patch.original_forward = module_to_patch.forward
                
                def new_forward(*args, m=module_to_patch, **kwargs):
                    def recompute_function(*args, **kwargs):
                        self.is_recomputing = True
                        try:
                            result = m.original_forward(*args, **kwargs)
                        finally:
                            self.is_recomputing = False
                        return result
                    return checkpoint(recompute_function, *args, use_reentrant=False, **kwargs)
                module_to_patch.forward = new_forward

    def _profiling_hook_factory(self, ref_counter):
        """一个通用的钩子工厂，用于在演练时填充引用计数器。"""
        def hook(module, inputs, outputs=None):
            param_ids = self.module_to_params_map.get(id(module), [])
            for param_id in param_ids:
                ref_counter[param_id] = ref_counter.get(param_id, 0) + 1
        return hook

    def profile_and_shard(self):
        """核心方法：执行分片和演练，返回所有客户端的初始化数据包。"""
        logger.info("[Coordinator] Starting parameter sharding and reference count profiling...")
        
        aggregation_mode = self.cfg.federate.get('aggregation_mode', 'gradient')
        use_offline_model = self.cfg.federate.get('use_offline_model', False)

        # --- 步骤 1: 参数分片 ---
        if aggregation_mode != 'local-reconstruct':
            all_param_shards = {rank: {} for rank in range(self.world_size)}
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if aggregation_mode == 'fl-sim':
                        # **不分片**，每个客户端都获得一个完整的副本
                        for rank in range(self.world_size):
                            all_param_shards[rank][name] = {
                                'shard_data': param.data.clone(), # 存储完整数据
                                'original_shape': param.shape,
                                'is_trainable': param.requires_grad,
                                'is_sharded': False # 明确标志为非分片
                            }
                    elif param.requires_grad and aggregation_mode == 'model':
                        # **不分片**，每个客户端都获得一个完整的副本
                        for rank in range(self.world_size):
                            all_param_shards[rank][name] = {
                                'shard_data': param.data.clone(), 
                                'original_shape': param.shape,
                                'is_trainable': True,
                                'is_sharded': False
                            }
                    else:
                        dim_to_split = 0
                        dim_size = param.shape[dim_to_split]
                        
                        # 替换 torch.chunk
                        if dim_size == 0:
                            shards = [torch.empty(0, dtype=param.dtype) for _ in range(self.world_size)]
                        elif dim_size < self.world_size:
                            # 规则不变：无法分给每个rank的，由rank 0持有
                            shards = [param.data.clone() if r == 0 else torch.empty(0, dtype=param.dtype) for r in range(self.world_size)]
                        else:
                            # 使用自定义的均匀分片逻辑
                            split_sizes = self._get_even_split_sizes(dim_size)
                            shards = torch.split(param.data, split_sizes, dim=dim_to_split)

                        for rank in range(self.world_size):
                            # 保存分片、原始形状和名称，这些信息客户端都需要
                            all_param_shards[rank][name] = {
                                'shard_data': shards[rank].clone(),
                                'original_shape': param.shape,
                                'is_trainable': param.requires_grad,
                                'is_sharded': True
                            }
            logger.info("[Coordinator] Parameter sharding complete.")

        if aggregation_mode == 'gradient':
            # --- 步骤 2: 演练引用计数 ---
            # 准备一个假的输入
            dummy_input_ids = torch.ones(1, self.cfg.llm.tok_len, dtype=torch.long, device=self.device)
            dummy_labels = torch.ones(1, self.cfg.llm.tok_len, dtype=torch.long, device=self.device)
            dummy_input = {'input_ids': dummy_input_ids, 'labels': dummy_labels}

            # 应用激活检查点 (如果配置了)
            if self.cfg.federate.use_activation_checkpointing:
                from federatedscope.llm.trainer.fedspeed_trainer import GPT2Block
                self._apply_checkpointing_for_profile(GPT2Block)

            # 注册演练钩子
            fwd_counter, recompute_counter, bwd_counter = {}, {}, {}
            
            # 使用一个技巧：在 post-forward 中区分是原始前向还是 recompute
            def post_fwd_profiling_hook(module, inputs, outputs):
                counter = recompute_counter if self.is_recomputing else fwd_counter
                param_ids = self.module_to_params_map.get(id(module), [])
                for pid in param_ids:
                    counter[pid] = counter.get(pid, 0) + 1

            fwd_handle = self.model.register_forward_hook(post_fwd_profiling_hook)
            bwd_handle = self.model.register_full_backward_hook(self._profiling_hook_factory(bwd_counter))

            # 执行演练
            logger.info("[Coordinator] Starting profiling forward pass...")
            self.model.train()
            outputs = self.model(**dummy_input)
            loss = outputs.loss
            logger.info("[Coordinator] Starting profiling backward pass...")
            loss.backward()
            logger.info("[Coordinator] Profiling complete.")

            # 清理钩子
            fwd_handle.remove()
            bwd_handle.remove()

            self.param_id_to_fwd_ref_count = fwd_counter
            self.param_id_to_recompute_ref_count = recompute_counter
            self.param_id_to_bwd_ref_count = bwd_counter

        # --- 步骤 3: 打包数据 ---
        all_client_packages = []
        ref_counts_package = {
            'fwd': self.param_id_to_fwd_ref_count,
            'recompute': self.param_id_to_recompute_ref_count,
            'bwd': self.param_id_to_bwd_ref_count,
            'id_to_param_map_keys': list(self.id_to_param_map.keys()) # 发送ID列表，客户端重建映射
        }
        
        if aggregation_mode == 'local-reconstruct':
            # 准备一个最终的、干净的 state_dict
            final_state_dict_to_send = OrderedDict()
            if not use_offline_model:
                model_to_pack = self.model
                # 遍历服务器模型的所有参数
                for name, param in model_to_pack.named_parameters():
                    # 归一化名称，移除所有可能的包装前缀，以得到纯净的 key
                    normalized_name = name
                    if normalized_name.startswith("base_model.model."):
                        normalized_name = normalized_name[len("base_model.model."):]
                    if normalized_name.startswith("model."):
                        normalized_name = normalized_name[len("model."):]
                    # 使用归一化的名称作为 key
                    final_state_dict_to_send[normalized_name] = param.cpu().clone()

            for rank in range(self.world_size):
                client_package = {
                    'ref_counts': ref_counts_package,
                    'full_model_state_dict': final_state_dict_to_send
                }
                all_client_packages.append(client_package)
        else:
            for rank in range(self.world_size):
                client_package = {
                    'ref_counts': ref_counts_package,
                    'shards': all_param_shards[rank]
                }
                all_client_packages.append(client_package)
        
        logger.info("[Coordinator] All client packages created successfully.")
        return all_client_packages

    def _get_even_split_sizes(self, dim_size):
        """
        计算每个 rank 的分片大小，确保尽可能均匀。
        返回一个列表，包含每个 rank 的分片大小。
        """
        if dim_size == 0:
            return [0] * self.world_size
        # 基础大小
        base_size = dim_size // self.world_size
        # 剩余的元素，分配给前面的 rank
        rem = dim_size % self.world_size
        # 每个rank的大小
        sizes = [base_size + 1 if i < rem else base_size for i in range(self.world_size)]
        return sizes


def natural_sort_key(s):
    """
    一个用于 sorted() 的 key 函数，可以实现自然排序 (e.g., 'h.10' 在 'h.2' 之后)。
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def log_memory_usage(stage_name):
    """打印当前 rank 的 GPU 内存使用情况。"""
    # if torch.cuda.is_available():
    #     allocated = torch.cuda.memory_allocated("cuda:0") / (1024 ** 2)  # MB
    #     reserved = torch.cuda.memory_reserved("cuda:0") / (1024 ** 2)    # MB
    #     max_allocated = torch.cuda.max_memory_allocated("cuda:0") / (1024 ** 2) # MB
    #     logger.info(
    #         f"[MemMon][{stage_name}] "
    #         f"Allocated: {allocated:.2f} MB | "
    #         f"Reserved: {reserved:.2f} MB | "
    #         f"Peak Allocated: {max_allocated:.2f} MB"
    #     )