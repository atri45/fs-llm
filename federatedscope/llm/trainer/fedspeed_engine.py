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
    def __init__(self, engine, name_to_param_map, sanitized_model_name: str):
        self.engine = engine
        self.rank = engine.rank
        self.device = engine.device
        self.name_to_param_map = name_to_param_map
        self.param_name_to_exec_idx = {}
        self.direction = 'forward'
        self.sanitized_model_name = sanitized_model_name

        # --- 配置 ---
        self.cache_path = engine.cfg.federate.get('model_cache_path')
        self.prefetch_depth = engine.cfg.federate.get('prefetch_depth', 2)
        self.gpu_mem_pressure_threshold = engine.cfg.federate.get('gpu_mem_pressure_threshold', 0.9)
        self.cpu_mem_pressure_threshold = engine.cfg.federate.get('cpu_mem_pressure_threshold', 0.9)
        self.model_specific_cache_path = os.path.join(self.cache_path, self.sanitized_model_name)

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

    def prime_cpu_cache(self, state_dict: dict):
        """用一个完整的 state_dict 填充 L2 (CPU) 缓存。"""
        self.cpu_cache.update(state_dict)

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
            param_cpu = self.cpu_cache[param_name]
            param_gpu = param_cpu.to(self.device, non_blocking=True)
            self._add_to_gpu_cache(param_name, param_gpu)
            return param_gpu
            
        # 3. L1 和 L2 都未命中，从磁盘加载
        key_to_find = param_name
        if key_to_find.startswith("base_model.model."):
            key_to_find = key_to_find[len("base_model.model."):]
        if key_to_find.startswith("model."):
            key_to_find = key_to_find[len("model."):]
        file_path = os.path.join(self.model_specific_cache_path, key_to_find + ".npy")
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
                if evicted_name in self.name_to_param_map:
                    param = self.name_to_param_map[evicted_name]
                    if param.fedspeed_status == "FULL_GPU_CACHED":
                        # 恢复其原始状态
                        if evicted_name in self.engine._my_persistent_shards:
                            param.fedspeed_status = "PARTIAL_PERSISTENT"
                        else:
                            param.fedspeed_status = "IN_CPU_CACHE"
                        # 清空其 data，表示不再是完整形态
                        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
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
            param = self.name_to_param_map[param_name]
            param.data = param_tensor_gpu
            param.fedspeed_status = "FULL_GPU_CACHED"

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

    def update_cache(self, param_name, shard_data, sharding_info, source_rank):
        """用收到的新分片更新 L1/L2 缓存中的完整参数。"""
        # 优先更新 L1 (GPU) 缓存
        if param_name in self.gpu_cache:
            full_param_gpu = self.gpu_cache[param_name]
            self._update_full_tensor_with_shard(full_param_gpu, shard_data, sharding_info, source_rank)

        # L1未命中，更新 L2 (CPU) 缓存
        elif param_name in self.cpu_cache:
            full_param_cpu = self.cpu_cache[param_name]
            self._update_full_tensor_with_shard(full_param_cpu, shard_data.cpu(), sharding_info, source_rank)

    def _update_full_tensor_with_shard(self, full_tensor, shard_data, sharding_info, source_rank):
        with torch.no_grad():
            rank_info = sharding_info['ranks_info'][source_rank]
            split_dim = sharding_info['split_dim']
            offset = rank_info['offset']
            size = rank_info['size']
            
            target_view = full_tensor.narrow(split_dim, offset, size)
            target_view.copy_(shard_data.to(full_tensor.device))

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
    客户端引擎,接收服务器预处理好的参数分片和引用计数，专注于执行训练循环。
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
        self._execution_order_built = False
        self._execution_order = []
        self._module_to_exec_idx = {}
        self.aggregation_timer = None
        self.sanitized_model_name = _sanitize_model_name_for_path(self.cfg.model.type)

        # 状态
        self.aggregation_mode = self.cfg.federate.get('aggregation_mode')
        self.aggregation_steps = self.cfg.federate.get('aggregation_steps', 1)
        self.local_cache_path = self.cfg.federate.get('model_cache_path')
        self.two_stage = self.cfg.federate.two_stages.get('use', False)
        self.stage_one_steps = self.cfg.federate.two_stages.get('stage_one_steps', 250)
        self.noniid = self.cfg.federate.two_stages.get('noniid', False)
        self.is_hybrid_mode = (
            self.aggregation_mode == 'shardedGradient' and
            self.two_stage and not self.noniid
        )
        if self.is_hybrid_mode:
            self.aggregation_mode = 'local-reconstruct'
        self.adaptive_weight_cfg = self.cfg.federate.get('adaptive_weight', {'use': False})
        self.is_in_stage_one = self.two_stage
        self.local_trainable_names_stage1 = None
        self.should_aggregate = False
        self._my_persistent_shards = {}
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
        if self.aggregation_steps > 1:
            self.remote_grad_accumulator = {}

        # 引用计数 (从服务器接收)
        self.param_id_to_fwd_ref_count = {}
        self.param_id_to_recompute_ref_count = {}
        self.param_id_to_bwd_ref_count = {}
        if ref_counts:
            self.param_id_to_fwd_ref_count = ref_counts['fwd']
            self.param_id_to_recompute_ref_count = ref_counts['recompute']
            self.param_id_to_bwd_ref_count = ref_counts['bwd']
        self.param_fwd_ref_count = {}
        self.param_recompute_ref_count = {}
        self.param_bwd_ref_count = {}

        # 映射关系
        self.module_to_params_map = {}
        self.id_to_param_map = {}
        self.name_to_param_map = {name: p for name, p in self.model.named_parameters()}

        # 存储分组元数据
        self.grouping_metadata = initial_shards.get('grouping_metadata', {})
        self.my_group_id = None
        self.my_layers_indices = []
        self.layer_to_peers_map = {}
        self.logical_layers_names = []
        self._param_to_layer_index_map = {} # 用于快速查找

        # --- 初始化引擎核心状态 ---
        if self.noniid and self.grouping_metadata:
            self.my_group_id = self.grouping_metadata['client_to_group_map'].get(self.rank)
            self.my_layers_indices = self.grouping_metadata['client_to_layers_map'].get(self.rank, [])
            self.layer_to_peers_map = self.grouping_metadata['layer_to_peers_map']
            self.logical_layers_names = self.grouping_metadata['logical_layers_names']
            # 构建一个参数名到层索引的快速查找映射
            self._build_param_to_layer_index_map()
        self._attach_module_ids()

        # 1. 获取元数据
        self.sharding_metadata = initial_shards.get('sharding_metadata', {})

        # 2. 创建 Reconstructor (即使在 In-Memory 模式下)
        #    因为它负责从磁盘加载数据，这是所有模式的共同起点。
        self.reconstructor = PipelinedReconstructor(self, self.name_to_param_map, self.sanitized_model_name)
        # self._prime_reconstructor_cpu_cache()
        self._initialize_with_reconstructor()

        # 3. 为所有模式设置参数的初始状态
        self._set_initial_param_status()

        # 3. 后续通用设置
        self.trainable_param_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
        self.ordered_trainable_names = sorted(list(self.trainable_param_names), key=natural_sort_key)
        if self.aggregation_mode == 'shardedGradient' or self.is_hybrid_mode:
            self.grad_shard_buffer = {name: [] for name in self.trainable_param_names}
            self.rank_to_client_id_map = initial_shards.get('rank_to_client_id_map')
        
        self._build_module_param_map()
        self._initialize_optimizer()
    
        initial_shards = None

     # --- 初始化函数 ---
    def _prime_reconstructor_cpu_cache(self):
        """
        一个统一的函数，负责从本地磁盘加载所有参数到 Reconstructor 的 CPU 缓存中。
        这是所有模式的共同第一步。
        """
        logger.info("Priming Reconstructor's CPU cache from local disk cache for all modes...")
        model_specific_cache_path = os.path.join(self.local_cache_path, self.sanitized_model_name)
        if not os.path.exists(model_specific_cache_path):
            raise FileNotFoundError(f"The required model cache path '{model_specific_cache_path}' does not exist.")
            
        offline_state_dict = {}
        for internal_name, _ in self.model.state_dict(return_trainable=False).items():
            
            # 用于查找文件的名称 (不带 'model.' 前缀)
            key_for_filename = internal_name
            if key_for_filename.startswith('model.'):
                key_for_filename = key_for_filename[len('model.'):]
            
            param_filename = key_for_filename + ".npy"
            file_path = os.path.join(model_specific_cache_path, param_filename)
            
            if os.path.exists(file_path):
                # 存入缓存时，使用模型内部的权威名称 (带前缀)
                offline_state_dict[internal_name] = torch.from_numpy(np.load(file_path))
            # 对于找不到的文件，不再打印警告，因为 _set_initial_param_status 会处理
        
        if not offline_state_dict:
             raise ValueError("No parameter files were found in the cache path.")
             
        self.reconstructor.prime_cpu_cache(offline_state_dict)
        logger.info(f"Reconstructor's CPU cache primed with {len(offline_state_dict)} parameters.")

    def _set_initial_param_status(self):
        """
        【最终版 v3】一个统一的函数，为所有模式的所有参数设置正确的初始状态。
        能够正确处理所有模式下的 LoRA 参数和基础参数。
        """
        # 1. 获取所有参数的权威 state_dict
        #    这既提供了权威的名称，也提供了对 param 对象的引用
        full_state_dict_with_refs = self.model.state_dict(return_trainable=False)
        
        # 2. 创建一个从名称到 param 对象的快速查找字典
        #    named_parameters() 的名称可能与 state_dict 不一致，我们需要一个映射
        param_map_from_named_params = {name: p for name, p in self.model.named_parameters()}
        
        with torch.no_grad():
            # 3. 遍历 state_dict 的 keys，这是我们的权威名称
            for name in full_state_dict_with_refs.keys():
                # 尝试从 named_parameters 中找到对应的 param 对象
                param = param_map_from_named_params.get(name)
                if param is None:
                    # state_dict 可能包含 buffer 等非 parameter，跳过它们
                    continue

                # --- 分模式设置状态 ---
                if self.aggregation_mode in ['local-reconstruct', 'shardedGradient']:
                    # --- Path A: Offloading 模式 ---
                    if self.aggregation_mode == 'shardedGradient':
                        if param.requires_grad: # 可训练参数 (LoRA)
                            param.fedspeed_status = "PARTIAL_PERSISTENT"
                        else: # 冻结参数
                            param.fedspeed_status = "IN_CPU_CACHE"
                    else: # local-reconstruct
                        if param.requires_grad:
                            param.fedspeed_status = "FULL_PERSISTENT"
                            # 将可训练参数 (LoRA) 直接加载到 GPU (使用其随机初始值)
                            param.data = param.data.to(self.device) 
                        else:
                            param.fedspeed_status = "IN_CPU_CACHE"
                
                else:
                    # --- Path B: In-Memory 模式 (model, fl-sim, gradient) ---
                    if param.requires_grad:
                        # --- 核心修复 ---
                        # 可训练参数 (LoRA) 是即时创建的，不需要从缓存加载。
                        # 它们本身就是完整的，应该持久化在 GPU。
                        param.fedspeed_status = "FULL_PERSISTENT"
                        # 确保它们在正确的设备上 (通常 get_llm 已经处理了)
                        param.data = param.data.to(self.device)
                        # --- 结束核心修复 ---
                    else:
                        # 冻结参数 (基础模型) 是分片的，需要网络重建
                        param.fedspeed_status = "SHARDED"
                        # 为其创建 CPU 上的分片副本
                        if self.reconstructor and name in self.reconstructor.cpu_cache:
                            full_param_cpu = self.reconstructor.cpu_cache[name]
                            
                            sharding_info = self.sharding_metadata.get(name)
                            if sharding_info:
                                my_rank_info = sharding_info['ranks_info'][self.rank]
                                offset, size = my_rank_info['offset'], my_rank_info['size']
                                shard_data = full_param_cpu.narrow(sharding_info['split_dim'], offset, size).clone()
                                param.fedspeed_shard = torch.nn.Parameter(shard_data, requires_grad=False)
                            else:
                                 param.fedspeed_shard = torch.nn.Parameter(torch.empty(0, dtype=param.dtype), requires_grad=False)
                        else:
                            logger.warning(f"Frozen param '{name}' not found in cache for sharding. Creating an empty shard.")
                            param.fedspeed_shard = torch.nn.Parameter(torch.empty(0, dtype=param.dtype), requires_grad=False)


                # 通用清理：确保不在 GPU 上的参数的 .data 为空
                if 'PERSISTENT' not in param.fedspeed_status:
                     if param.data.numel() > 0:
                        param.data = torch.empty(0, dtype=param.dtype, device=self.device)

    def _initialize_with_reconstructor(self):
        """
        为 Offloading 模式 (`local-reconstruct`, `shardedGradient`) 进行初始化。
        【最终版：确保 LoRA 参数被添加到 CPU 缓存】
        """
        # 1. 创建 Reconstructor
        self.reconstructor = PipelinedReconstructor(self, self.name_to_param_map, self.sanitized_model_name)
        
        # 2. 从本地磁盘加载【基础模型】数据到 Reconstructor 的 CPU 缓存
        self._prime_reconstructor_cpu_cache() # 这个函数现在只负责从磁盘加载

        # --- START OF FINAL FIX ---
        # 3. 【关键】对于 shardedGradient 模式，将内存中新创建的 LoRA 参数手动添加到 CPU 缓存
        if self.aggregation_mode == 'shardedGradient':
            logger.info("Injecting newly created LoRA parameters into the CPU cache for shardedGradient mode...")
            num_injected = 0
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad: # 识别可训练参数 (LoRA)
                        param.fedspeed_status = "PARTIAL_PERSISTENT"
                        if name not in self.reconstructor.cpu_cache:
                            # 将参数的 .data 复制到CPU，然后存入缓存
                            self.reconstructor.cpu_cache[name] = param.data.cpu().clone()
                            num_injected += 1
            logger.info(f"Injected {num_injected} LoRA parameters into the CPU cache.")

    def _build_param_to_layer_index_map(self):
        """
        构建一个从参数全名到其所属逻辑层索引的映射。
        """
        if not self.logical_layers_names:
            return
        
        # 创建一个层名称到索引的映射
        layer_name_to_idx = {name: i for i, name in enumerate(self.logical_layers_names)}
        sorted_layer_names = sorted(self.logical_layers_names, key=len, reverse=True)

        for param_name, param in self.model.named_parameters():
            # 遍历排序后的层名称
            for layer_name in sorted_layer_names:
                # 1. 检查是否是前缀
                if param_name.startswith(layer_name):
                    # 2. 检查前缀后面的字符是否是 '.' 或字符串结尾
                    #    这可以防止 'h.1' 错误地匹配 'h.10'
                    if len(param_name) == len(layer_name) or param_name[len(layer_name)] == '.':
                        layer_idx = layer_name_to_idx.get(layer_name)
                        if layer_idx is not None:
                            self._param_to_layer_index_map[param_name] = layer_idx
                            break # 找到最精确的匹配后就跳出
        logger.info(f"Built parameter-to-layer-index map for {len(self._param_to_layer_index_map)} parameters.")

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

    def _initialize_optimizer(self):
        """
        根据 aggregation_mode 初始化相应的优化器和参数缓冲区。
        """        
        # 遍历一次所有参数，附加通用属性并构建 id_to_param_map
        self.id_to_param_map.clear()
        for name, param in self.model.named_parameters():
            param.fedspeed_name = name
            param.fedspeed_original_shape = param.shape
            self.id_to_param_map[id(param)] = param

        # --- 根据模式进行不同的初始化 ---
        if self.aggregation_mode == 'gradient':
            self._initialize_for_gradient_aggregation()
        elif self.aggregation_mode in ['model', 'local-reconstruct', 'fl-sim']:
            self._initialize_for_model_aggregation()
        elif self.aggregation_mode == 'shardedGradient':
            self._initialize_for_shardedGradient()
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

    def _initialize_for_shardedGradient(self):
        """为 'shardedGradient' 模式初始化分片优化器。"""
        logger.info(f"Rank {self.rank}: Initializing for 'shardedGradient' mode.")
        
        my_shard_param_views = []

        if not self.sharding_metadata:
            logger.error("Engine's sharding_metadata is EMPTY. Cannot assign shards.")

        for name in self.ordered_trainable_names:
            param = self.name_to_param_map.get(name)
            if param is None: continue
            # 从元数据中找到我负责的分片信息
            param_sharding_info = self.sharding_metadata.get(name)
            if not param_sharding_info:
                logger.warning(f"Sharding metadata not found for trainable param '{name}'. Skipping.")
                continue

            my_rank_info = param_sharding_info['ranks_info'].get(self.rank)
            if my_rank_info and my_rank_info['size'] > 0:

                is_lora_param = 'lora_' in name.lower()

                if is_lora_param:
                    # 1. 路径 A: LoRA 参数 (即时创建的随机权重)
                    # 从模型实例的 .data 属性获取完整的参数张量 (这是随机初始化的值)
                    # 确保它在 GPU 上
                    full_param_source = param.data.to(self.device).clone() 
                    
                elif name in self.reconstructor.cpu_cache:
                    # 2. 路径 B: 非 LoRA 可训练参数 (例如，Bias，或者全参数训练中的所有参数)
                    # 从 Reconstructor 的 CPU 缓存中获取完整的参数张量
                    full_param_cpu = self.reconstructor.cpu_cache[name]
                    # 确保数据被复制到 GPU
                    full_param_source = full_param_cpu.to(self.device)
                else:
                    logger.error(f"Full parameter for '{name}' not found in CPU cache or is not a LORA param. Cannot create shard.")
                    continue
                
                # --- 核心切分逻辑 ---
                split_dim = param_sharding_info['split_dim']
                offset = my_rank_info['offset']
                size = my_rank_info['size']
                
                # 从完整参数张量中创建一个视图(view)作为我的分片
                shard_data_gpu_view = full_param_source.narrow(split_dim, offset, size)
                
                # 将这个视图变成一个需要梯度的 Parameter 对象，以便优化器管理
                # 需要 clone() 避免视图问题
                shard_param_for_optim = torch.nn.Parameter(shard_data_gpu_view.clone())
                shard_param_for_optim.fedspeed_name = name 
                
                my_shard_param_views.append(shard_param_for_optim)
                self._my_persistent_shards[name] = shard_param_for_optim

        if not my_shard_param_views:
            logger.warning(f"Rank {self.rank}: No parameter shards assigned to this client. Optimizer will not be created.")
            self.local_optimizer = None
            return

        # 创建一个只作用于这些分片视图的优化器
        self.local_optimizer = get_optimizer(my_shard_param_views, **self.cfg.train.optimizer)
        logger.info(f"Rank {self.rank}: Sharded optimizer created, managing {len(my_shard_param_views)} parameter shards.")

        # 如果开启了两阶段，则确定第一阶段要训练的参数【名称】
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
        self.layer_idx_to_owner_rank_map = {}
        for owner_rank, indices_tensor in enumerate(block_chunks):
            for layer_idx in indices_tensor.tolist():
                self.layer_idx_to_owner_rank_map[layer_idx] = owner_rank
        logger.info(f"Built layer-to-owner-rank map: {self.layer_idx_to_owner_rank_map}")
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

        # 1. 动态构建执行顺序
        if self.model.training and not self._execution_order_built and self.reconstructor:
            if list(module.parameters(recurse=False)):
                module_id = module.fedspeed_id
                if module_id not in self._module_to_exec_idx:
                    self._module_to_exec_idx[module_id] = len(self._execution_order)
                    self._execution_order.append(module)

        with torch.no_grad():
            # 遍历模块的【所有】参数，检查它们的状态
            for name, param in module.named_parameters(recurse=False):
                if param is not None and hasattr(param, 'fedspeed_status'):
                    if param.fedspeed_status == "FULL_GPU_CACHED":
                        continue
                    elif param.fedspeed_status == "SHARDED":
                        self._network_reconstruct_param(param)
                    elif param.fedspeed_status == "IN_CPU_CACHE":
                        self._local_reconstruct_param(param)
                    elif param.fedspeed_status == "PARTIAL_PERSISTENT":
                        if not self.reconstructor:
                            logger.error("LocalReconstructor not initialized. Cannot perform local reconstruction.")
                            return
                        param_name = param.fedspeed_name
                        logger.debug(f"load param from cache: {param_name}")
                        full_param_gpu = self.reconstructor.load_param(param_name)
                        my_latest_shard = self._my_persistent_shards[param_name]
                        
                        my_rank_info = self.sharding_metadata[param_name]['ranks_info'][self.rank]
                        split_dim = self.sharding_metadata[param_name]['split_dim']
                        offset = my_rank_info['offset']
                        size = my_rank_info['size']

                        # 直接在加载到 GPU 的完整参数上进行就地修正
                        full_param_gpu.narrow(split_dim, offset, size).copy_(my_latest_shard)
                       
                        param.data = full_param_gpu

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
        if not self.model.training:
            # 如果是评估模式 (model.eval())，则不执行任何参数释放或预取
            return
    
        logger.debug(f"[POST-FWD HOOK @ Rank {self.rank}] For {type(module).__name__}")
        
        # 1. 参数释放
        if self.aggregation_mode == 'model':
            self._release_all_full_params()
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
        if self.aggregation_mode in ['local-reconstruct', 'shardedGradient']:
            self.reconstructor.direction = 'forward'
            # a. 获取接下来要预取的模块 (方向是 forward)
            next_modules = self._get_modules_for_prefetch(module, is_forward=True)
            # b. 调度预取
            self._schedule_prefetch_for_modules(next_modules)

    def _get_modules_for_prefetch(self, current_module, is_forward):
        """
        根据当前模块和传播方向，获取接下来需要预取的模块。
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
                if p is not None and hasattr(p, 'fedspeed_status'):
                    if p.fedspeed_status in ["ON_DISK", "IN_CPU_CACHE", "PARTIAL_PERSISTENT"]:
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
        if not self.model.training:
            # 如果是评估模式 (model.eval())，则不执行任何参数释放或预取
            return
            
        logger.debug(f"[POST-BWD HOOK @ Rank {self.rank}] For {type(module).__name__}")

        # 1. 参数释放
        if self.aggregation_mode == 'model':
            self._release_all_full_params()
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
        if self.aggregation_mode in ['local-reconstruct', 'shardedGradient']:
            self.reconstructor.direction = 'backward'
            # a. 获取接下来要预取的模块 (方向是 forward)
            next_modules = self._get_modules_for_prefetch(module, is_forward=False)
            # b. 调度预取
            self._schedule_prefetch_for_modules(next_modules)

    def _collect_grad_hook_factory(self, name):
        def hook(grad):
            if grad is not None:
                self.temp_full_gradients[name] = grad.detach()
        return hook
    
    # --- 核心训练方法 ---

    def forward(self, *args, **kwargs):    
        logger.debug(f"Rank {self.rank}: Starting forward pass...")

        # 加载预先计算好的计数
        self.param_fwd_ref_count = self.param_id_to_fwd_ref_count.copy()

        # 前向传播
        outputs = self.model(*args, **kwargs)

        # 在第一次前向传播结束后，固化执行顺序
        if self.reconstructor and self.model.training and not self._execution_order_built:
            self._execution_order_built = True
            param_name_to_exec_idx = {}
            for idx, module in enumerate(self._execution_order):
                for p_name, p in module.named_parameters(recurse=False):
                    if hasattr(p, 'fedspeed_name'):
                        param_name_to_exec_idx[p.fedspeed_name] = idx
            self.reconstructor.param_name_to_exec_idx = param_name_to_exec_idx
            logger.info(f"Dynamically built execution order with {len(self._execution_order)} modules.")

            # 3. 如果开启了两阶段，则确定第一阶段要训练的参数【名称】
            # if self.two_stage:
            #     self._partition_trainable_params_for_stage1()

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
        if self.aggregation_mode != 'gradient':
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
        根据 self.aggregation_mode 选择不同的更新路径。
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
                self.is_in_stage_one = False
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
            self.aggregation_timer.start()
            trainable_params = self.local_optimizer.param_groups[0]['params']
            params_to_send_vec = parameters_to_vector([p.data for p in trainable_params]).clone()
            self.aggregation_timer.stop()

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
            self.aggregation_timer.start()
            trainable_params = self.local_optimizer.param_groups[0]['params']
            params_to_send_vec = parameters_to_vector([p.data for p in trainable_params]).clone()
            self.aggregation_timer.stop()
            
            # 清理未使用的缓存内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return params_to_send_vec

        # 4. 清理未使用的缓存内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return None

    def sharded_step(self, my_local_grad_shards: dict):
        """
        在 'shardedGradient' 模式下执行一步优化。
        1. 对每个分片，聚合本地计算的梯度和收到的梯度 (FedAvg)。
        2. 调用分片优化器更新。
        3. 返回更新后的参数分片信息，用于广播。
        """
        if not self.local_optimizer:
            return [], False

        # --- 阶段一：层聚合与梯度掩码 ---
        if self.noniid and self.is_in_stage_one:
            # 1. 聚合收到的层梯度 (FedAvg)
            all_grads_for_update = {name: [] for name in self.ordered_trainable_names}
            
            # a. 加入自己本地计算的梯度
            for name, grad in my_local_grad_shards.items():
                all_grads_for_update[name].append((grad, self.current_step))

            # b. 从缓冲区中加入接收到的梯度
            for name, received_shards in self.grad_shard_buffer.items():
                all_grads_for_update[name].extend(received_shards)

            # 2. 执行 FedAvg 并应用梯度掩码
            with torch.no_grad():
                for param_shard_view in self.local_optimizer.param_groups[0]['params']:
                    param_name = param_shard_view.fedspeed_name
                    
                    # 检查这个参数是否属于我负责的层
                    layer_idx = self._param_to_layer_index_map.get(param_name)
                    if layer_idx not in self.my_layers_indices:
                        # 【梯度掩码】不属于我，梯度设为 0
                        param_shard_view.grad = torch.zeros_like(param_shard_view.data)
                        continue
                    
                    grad_tuples = all_grads_for_update.get(param_name, [])
                    if not grad_tuples:
                        param_shard_view.grad = torch.zeros_like(param_shard_view.data)
                        continue
                    
                    self.aggregation_timer.start()
                    num_grads_to_agg = len(grad_tuples)
                    grad_steps = [t[1] for t in grad_tuples]
                    logger.debug(
                        f"[Stage 1 Aggregation] Param '{param_name[:30]}...': "
                        f"Aggregating {num_grads_to_agg} grads from steps: {grad_steps}"
                    )

                    # --- 在阶段一执行加权平均 ---
                    weighted_grad_sum = torch.zeros_like(grad_tuples[0][0], device=self.device)
                    total_weight = 0.0
                    for grad, r_step in grad_tuples:
                        alpha = self._calculate_adaptive_alpha(r_step)
                        weighted_grad_sum.add_(grad.to(self.device), alpha=alpha)
                        total_weight += alpha
                    
                    if total_weight > 1e-6:
                        avg_grad = weighted_grad_sum / total_weight
                    else:
                        avg_grad = torch.zeros_like(param_shard_view.data)

                    # 阶段一的梯度是完整的，需要切分出我的分片
                    sharding_info = self.sharding_metadata[param_name]
                    my_rank_info = sharding_info['ranks_info'][self.rank]
                    offset, size = my_rank_info['offset'], my_rank_info['size']
                    
                    avg_grad_shard = avg_grad.narrow(sharding_info['split_dim'], offset, size)
                    param_shard_view.grad = avg_grad_shard.to(self.device)

                    self.aggregation_timer.stop()

                    # d. 清理缓冲区
                    if param_name in self.grad_shard_buffer:
                        self.grad_shard_buffer[param_name] = []

                # 3. 更新模型
                # # 梯度裁剪
                # params_with_grad = [p for p in self.local_optimizer.param_groups[0]['params'] if p.grad is not None]
                # if params_with_grad:
                #     clip_grad_norm_(params_with_grad, max_norm=1.0)
                self.local_optimizer.step()
                self.local_optimizer.zero_grad()

                # 阶段一不广播参数，返回空值
                return [], False
        # --- 阶段二 ---
        else:
            # 优化器更新
            updated_shards_for_broadcast = []
            model_was_updated_by_peers = False  

            # 遍历由我的分片优化器管理的每个参数分片
            for param_shard_view in self.local_optimizer.param_groups[0]['params']:
                param_name = param_shard_view.fedspeed_name
                
                # 1. 聚合梯度 (FedAvg)
                # a. 获取本地计算出的梯度分片
                local_grad_shard = my_local_grad_shards.get(param_name)
                if local_grad_shard is None:
                    # 只有在缓冲区也没有梯度时才跳过
                    if not self.grad_shard_buffer.get(param_name, []):
                        logger.warning(f"No local or remote grad shard for '{param_name}'. Skipping update.")
                        continue
                    # 如果只有远程梯度，创建一个零张量作为本地梯度
                    local_grad_shard = torch.zeros_like(param_shard_view.data)
                    
                # b. 获取从网络接收的梯度分片
                received_grad_tuples = self.grad_shard_buffer.get(param_name, [])
                
                # c. 将所有梯度分片（本地的+接收的）聚合
                if received_grad_tuples:
                    self.aggregation_timer.start()
                    model_was_updated_by_peers = True
                    num_remote_grads = len(received_grad_tuples)
                    remote_steps = [t[1] for t in received_grad_tuples]
                    logger.debug(
                        f"[Stage 2 Aggregation] ParamShard '{param_name[:30]}...': "
                        f"Aggregating local grad (step {self.current_step}) with {num_remote_grads} remote grads from steps: {remote_steps}"
                    )
                    alpha_local = self._calculate_adaptive_alpha(self.current_step)
                    weighted_grad_sum = local_grad_shard.to(self.device) * alpha_local
                    total_weight = alpha_local

                    for grad_shard, received_step in received_grad_tuples:
                        alpha_remote = self._calculate_adaptive_alpha(received_step)
                        weighted_grad_sum.add_(grad_shard.to(self.device), alpha=alpha_remote)
                        total_weight += alpha_remote
                    
                    if total_weight > 1e-6:
                        avg_grad_shard = weighted_grad_sum / total_weight
                    else:
                        avg_grad_shard = torch.zeros_like(param_shard_view.data)
                    param_shard_view.grad = avg_grad_shard
                    self.aggregation_timer.stop()
                else:
                    param_shard_view.grad = local_grad_shard.to(self.device)

                # d. 清理缓冲区
                if param_name in self.grad_shard_buffer:
                    self.grad_shard_buffer[param_name] = []

            # 梯度裁剪
            # params_with_grad = [p for p in self.local_optimizer.param_groups[0]['params'] if p.grad is not None]
            # if params_with_grad:
            #     clip_grad_norm_(params_with_grad, max_norm=1.0)
            self.local_optimizer.step()
            self.local_optimizer.zero_grad()
            self._write_back_persistent_shards_to_gpu_cache()

            # 准备广播更新后的参数分片
            self.aggregation_timer.start()
            with torch.no_grad():
                for param_shard_view in self.local_optimizer.param_groups[0]['params']:
                    broadcast_content = {
                        'param_name': param_shard_view.fedspeed_name,
                        'shard_data': param_shard_view.data.cpu().clone(),
                        'source_rank': self.rank,
                        'step': self.current_step
                    }
                    updated_shards_for_broadcast.append(broadcast_content)
            self.aggregation_timer.stop()

            return updated_shards_for_broadcast, model_was_updated_by_peers

    def sync_model_layers_via_broadcast(self):
        """
        通过逐层广播的方式，同步在阶段一由不同客户端训练的参数层。
        """
        logger.info(f"Rank {self.rank}: Starting layer-wise model synchronization via Broadcast...")

        # 确保我们有用于阶段一的完整优化器
        optimizer = self.local_optimizer
            
        if not optimizer:
            logger.error("Cannot perform layer-wise sync: No valid full-parameter optimizer found.")
            return
        
        # 获取所有可训练参数的 map
        trainable_params_map = {p.fedspeed_name: p for p in optimizer.param_groups[0]['params']}
        
        with torch.no_grad():
            # 1. 确定每个客户端负责的层 (这个逻辑需要从 _partition_trainable_params_for_stage1 中获取)
            #    我们假设 _partition_... 函数已经被调用，并且 self.local_trainable_names_stage1 已经填充
            
            # 2. 遍历所有逻辑层
            for layer_idx, layer_name in enumerate(self.logical_layers_names):
                # a. 确定这一层的广播源 (src_rank)
                src_rank = self.layer_idx_to_owner_rank_map.get(layer_idx)
                if src_rank is None:
                    logger.warning(f"Could not find owner rank for layer {layer_idx}. Skipping sync for this layer.")
                    continue
                
                # b. 将该层的所有参数打包成一个向量
                layer_params = []
                for name, p in self.name_to_param_map.items():
                    if self._param_to_layer_index_map.get(name) == layer_idx and p.requires_grad:
                        # 确保我们使用的是优化器中管理的那个最新的参数对象
                        if name in trainable_params_map:
                             layer_params.append(trainable_params_map[name])

                if not layer_params:
                    continue

                # 创建一个正确大小的缓冲区来发送/接收
                vec_size = sum(p.numel() for p in layer_params)
                if vec_size == 0: continue
                layer_vec = torch.empty(vec_size, dtype=layer_params[0].dtype, device=self.device)
                
                # c. 如果我是广播源，我填充缓冲区
                if self.rank == src_rank:
                    # 从我的最新参数中填充
                    vector_to_parameters(parameters_to_vector([p.data for p in layer_params]), [layer_vec]) # 这是一个小技巧
                
                # d. 执行广播
                dist.broadcast(layer_vec, src=src_rank, group=self.comm_group)

                # e. 【所有客户端】用收到的向量来更新自己的相应参数层
                vector_to_parameters(layer_vec, layer_params)

            # 3. 将同步后的最新参数，也写回到 CPU 缓存
            if self.reconstructor:
                logger.info(f"Rank {self.rank}: Updating CPU cache with layer-wise synchronized parameters.")
                for param in optimizer.param_groups[0]['params']:
                    if param.fedspeed_name in self.reconstructor.gpu_cache:
                        self.reconstructor.gpu_cache[param.fedspeed_name].copy_(param.data.cpu())
                    if param.fedspeed_name in self.reconstructor.cpu_cache:
                        self.reconstructor.cpu_cache[param.fedspeed_name].copy_(param.data.cpu())
        
        logger.info(f"Rank {self.rank}: Layer-wise model synchronization complete.")

    def sync_model_at_stage_transition(self):
        """
        在阶段一结束时，执行全局模型同步。
        每个客户端贡献出自己负责的最新参数层，然后大家通过 all-gather 重建完整模型。
        """
        if not (self.noniid and self.is_in_stage_one):
            logger.warning("sync_model_at_stage_transition called but not in stage one of grouping mode.")
            return

        logger.info(f"Rank {self.rank}: Starting global model synchronization at stage transition...")

        with torch.no_grad():
            # 遍历所有逻辑层
            for layer_idx, layer_name in enumerate(self.logical_layers_names):
                # 找到这一层的所有参数
                layer_params = [
                    self.name_to_param_map[name]
                    for name in self.ordered_trainable_names
                    if self._param_to_layer_index_map.get(name) == layer_idx
                ]

                if not layer_params:
                    logger.debug(f"No trainable parameters found for layer {layer_idx} ('{layer_name}'). Skipping sync.")
                    continue
                
                # 确定谁是这一层的“源” (source of truth)
                # 理论上，layer_to_peers_map[layer_idx] 里的所有人都拥有最新版本
                # 我们选择 rank 最小的那个作为广播源 (src_rank)
                src_rank = min(self.layer_to_peers_map[layer_idx])
                
                # 计算正确的向量大小
                vec_size = sum(p.numel() for p in layer_params)
                if vec_size == 0: continue

                # 创建发送/接收缓冲区
                layer_vec = torch.empty(vec_size, dtype=layer_params[0].dtype, device=self.device)
                
                if self.rank == src_rank:
                    # a. 在本地重组出这一层【完整】的、最新的参数
                    reconstructed_layer_params = []
                    for param in layer_params:
                        name = param.fedspeed_name
                        
                        # i. 从 CPU 缓存获取完整的、微过时的参数
                        if name not in self.reconstructor.cpu_cache:
                            logger.error(f"Param '{name}' not found in CPU cache during sync. Using zeros.")
                            reconstructed_layer_params.append(torch.zeros_like(param, device=self.device))
                            continue
                        
                        full_param_cpu = self.reconstructor.cpu_cache[name].clone()
                        
                        # ii. 用自己最新的持久化分片去修正它
                        if name in self._my_persistent_shards:
                            my_latest_shard = self._my_persistent_shards[name].data.cpu()
                            sharding_info = self.sharding_metadata[name]
                            my_rank_info = sharding_info['ranks_info'][self.rank]
                            offset, size = my_rank_info['offset'], my_rank_info['size']
                            full_param_cpu.narrow(sharding_info['split_dim'], offset, size).copy_(my_latest_shard)
                        
                        reconstructed_layer_params.append(full_param_cpu.to(self.device))
                        
                    # b. 将重组后的【完整】参数列表，打包到 layer_vec 中
                    source_vec = parameters_to_vector(reconstructed_layer_params)
                    layer_vec.copy_(source_vec) # 使用 copy_ 填充
                
                # 执行广播
                dist.broadcast(layer_vec, src=src_rank, group=self.comm_group)

                # 所有客户端都用收到的向量来更新自己的【持久化分片】和【CPU缓存】
                # 1. 恢复出完整的、更新后的层参数
                temp_layer_params_data = []
                pointer = 0
                for param in layer_params:
                    num_param = param.numel()
                    param_data = layer_vec[pointer : pointer + num_param].view_as(param)
                    temp_layer_params_data.append(param_data)
                    pointer += num_param

                # 2. 遍历这些更新后的层参数，更新自己系统中的相应部分
                for i, param in enumerate(layer_params):
                    updated_full_param_data = temp_layer_params_data[i]
                    name = param.fedspeed_name
                    
                    # a. 更新我的持久化分片 (如果我负责这个参数)
                    if name in self._my_persistent_shards:
                        sharding_info = self.sharding_metadata.get(name)
                        if not sharding_info:
                            logger.warning(f"Sharding metadata for '{name}' not found during sync. Skipping shard update.")
                            continue
                        my_rank_info = sharding_info['ranks_info'][self.rank]
                        offset, size = my_rank_info['offset'], my_rank_info['size']
                        
                        # 从更新后的完整参数中切出我的新分片
                        new_shard_data = updated_full_param_data.narrow(sharding_info['split_dim'], offset, size)
                        self._my_persistent_shards[name].data.copy_(new_shard_data)

                    # b. 更新 CPU 缓存
                    if self.reconstructor and name in self.reconstructor.cpu_cache:
                        self.reconstructor.cpu_cache[name].copy_(updated_full_param_data.cpu())

        dist.barrier()
        logger.info(f"Rank {self.rank}: Global model synchronization complete.")
        self.is_in_stage_one = False

    def _write_back_persistent_shards_to_gpu_cache(self):
        """
        将更新后的持久化分片，写回到 Reconstructor 的 GPU 缓存中。
        这是确保模型更新能被下一次迭代使用的关键步骤。
        """
        # 检查 Reconstructor 和它的 GPU 缓存是否存在
        if not hasattr(self, 'reconstructor') or not self.reconstructor.gpu_cache:
            return

        with torch.no_grad():
            # 遍历所有我负责的、刚刚被 optimizer 更新的持久化分片
            for name, persistent_shard_param in self._my_persistent_shards.items():
                
                # 检查这个参数的完整版本是否【当前】就在 GPU 缓存中
                if name in self.reconstructor.gpu_cache:
                    # 获取缓存中的完整参数张量
                    full_param_gpu_cached = self.reconstructor.gpu_cache[name]
                    
                    # 获取分片所需的信息
                    sharding_info = self.sharding_metadata.get(name)
                    if not sharding_info: continue # 防御性检查

                    split_dim = sharding_info['split_dim']
                    my_rank_info = sharding_info['ranks_info'][self.rank]
                    offset, size = my_rank_info['offset'], my_rank_info['size']
                    
                    # 定位到缓存中完整张量的相应区域
                    target_view = full_param_gpu_cached.narrow(split_dim, offset, size)
                    
                    # 用我最新的持久化分片数据，【就地更新】缓存中的张量
                    target_view.copy_(persistent_shard_param.data)
                    
                    logger.debug(f"Rank {self.rank}: Wrote back updated shard for '{name}' to GPU cache.")

    def process_and_dispatch_gradients(self):
        """
        拆分本地计算的完整梯度，保留自己的分片，并返回待发送给其他人的分片。
        """
        if not self.temp_full_gradients:
            return {}, []

        # --- 阶段一：分组跨组层聚合 ---
        if self.noniid and self.is_in_stage_one:
            my_local_full_gradients = {name: grad.clone() for name, grad in self.temp_full_gradients.items()}
            
            # 2. 准备分发的梯度（只在聚合步）
            grouped_shards_to_dispatch = {}
            
            # 检查是否是聚合步
            should_aggregate_now = (self.current_step) % self.aggregation_steps == 0
            
            if should_aggregate_now:
                self.aggregation_timer.start()
                logger.debug(f"[Stage 1] Aggregation step. Preparing to dispatch layer gradients.")
                # 将梯度按层分组
                grads_by_layer = {i: {} for i in range(len(self.logical_layers_names))}
                for name, grad in self.temp_full_gradients.items():
                    layer_idx = self._param_to_layer_index_map.get(name)
                    if layer_idx is not None:
                        grads_by_layer[layer_idx][name] = grad

                # 按层准备要发送的消息
                for layer_idx, layer_grads_dict in grads_by_layer.items():
                    if not layer_grads_dict: continue
                    
                    # 找到通信同伴
                    peer_ranks = self.layer_to_peers_map.get(layer_idx, [])
                    if layer_idx not in self.my_layers_indices:
                        logger.debug(f"    - Skipping layer {layer_idx} as it's not my responsibility.")
                        continue
                    for target_rank in peer_ranks:
                        if target_rank == self.rank: continue
                        
                        target_client_id = self.rank_to_client_id_map.get(target_rank)
                        if not target_client_id: continue

                        if target_client_id not in grouped_shards_to_dispatch:
                            grouped_shards_to_dispatch[target_client_id] = []
                        
                        # 打包整个层的梯度
                        for name, grad in layer_grads_dict.items():
                            dispatch_content = {
                                'param_name': name,
                                'grad_shard_data': grad.cpu().clone(),
                                'source_rank': self.rank,
                                'step': self.current_step
                            }
                            grouped_shards_to_dispatch[target_client_id].append(dispatch_content)
                self.aggregation_timer.stop()

            self.temp_full_gradients.clear()

            # 返回完整的本地梯度 和 (可能为空的)待分发梯度
            return my_local_full_gradients, grouped_shards_to_dispatch
        # --- 阶段二或不分组 ---
        else:
            my_local_grad_shards = {}
            grouped_shards_to_dispatch = {}

            # 检查是否需要聚合
            self.should_aggregate = False
            if self.two_stage:
                if self.current_step == self.stage_one_steps:
                    # 阶段转换点，强制聚合
                    self.should_aggregate = True
                    self.is_in_stage_one = False
                elif self.current_step > self.stage_one_steps and self.current_step % self.aggregation_steps == 0:
                    # 第二阶段，按频率聚合
                    self.should_aggregate = True
                # else: 第一阶段，不聚合
            else: # 非两阶段模式
                if self.current_step % self.aggregation_steps == 0:
                    self.should_aggregate = True

            for name, full_grad in self.temp_full_gradients.items():
                param_sharding_info = self.sharding_metadata.get(name)
                if not param_sharding_info:
                    continue

                split_dim = param_sharding_info['split_dim']
                
                split_sizes = [info['size'] for rank, info in sorted(param_sharding_info['ranks_info'].items())]
                grad_shards_list = torch.split(full_grad, split_sizes, dim=split_dim)

                for rank_idx, grad_shard in enumerate(grad_shards_list):
                    if rank_idx == self.rank:
                        my_local_grad_shards[name] = grad_shard
                    else:
                        # 这是需要发给别人的分片
                        if self.should_aggregate:
                            if self.aggregation_steps > 1:
                                # 多步模式：存入累加器
                                target_rank = rank_idx
                                if target_rank not in self.remote_grad_accumulator:
                                    self.remote_grad_accumulator[target_rank] = {}
                                if name not in self.remote_grad_accumulator[target_rank]:
                                    self.remote_grad_accumulator[target_rank][name] = {
                                        'sum': torch.zeros_like(grad_shard), 'count': 0,
                                        'step': self.current_step
                                        }
                                self.remote_grad_accumulator[target_rank][name]['sum'].add_(grad_shard)
                                self.remote_grad_accumulator[target_rank][name]['count'] += 1
                                self.remote_grad_accumulator[target_rank][name]['step'] = self.current_step
                            else:
                                self.aggregation_timer.start()
                                dispatch_content = {
                                    'param_name': name,
                                    'grad_shard_data': grad_shard.cpu().clone(),
                                    'source_rank': self.rank,
                                    'step': self.current_step
                                }
                                target_client_id = self.rank_to_client_id_map.get(rank_idx)

                                # 将这个分片内容添加到对应目标rank的列表中
                                if target_client_id not in grouped_shards_to_dispatch:
                                    grouped_shards_to_dispatch[target_client_id] = []
                                grouped_shards_to_dispatch[target_client_id].append(dispatch_content)
                                self.aggregation_timer.stop()
            
            self.temp_full_gradients.clear()

            if self.should_aggregate and self.aggregation_steps > 1:
                self.aggregation_timer.start()
                grouped_shards_to_dispatch = self._prepare_dispatch_from_accumulator()
                self.aggregation_timer.stop()

            # 返回本地分片 和 按目标分组的待发送分片
            return my_local_grad_shards, grouped_shards_to_dispatch

    def _prepare_dispatch_from_accumulator(self):
        """
        从累加器中准备要分发的平均梯度。
        在聚合 step 调用此函数。
        """
        if not self.remote_grad_accumulator:
            return {}
        
        grouped_shards_to_dispatch = {}

        # 遍历累加器中的每个目标 rank
        for target_rank, param_grads in self.remote_grad_accumulator.items():
            
            # 为这个目标 rank 准备一个分片列表
            shard_list_for_target = []
            
            # 遍历这个 rank 对应的所有参数的累加梯度
            for name, acc_data in param_grads.items():
                if acc_data['count'] > 0:
                    # 1. 计算平均梯度
                    #    使用 in-place 除法以节省内存
                    avg_grad = acc_data['sum'].div_(acc_data['count'])
                    
                    # 2. 打包 content
                    content = {
                        'param_name': name,
                        'grad_shard_data': avg_grad.cpu(), # 移动到 CPU 准备发送
                        'source_rank': self.rank,
                        'step': acc_data['step']
                    }
                    shard_list_for_target.append(content)
            
            # 3. 将这个目标的整个分片列表，用 client_id 作为 key 存入最终的分组字典
            if shard_list_for_target:
                target_client_id = self.rank_to_client_id_map.get(target_rank)
                if target_client_id:
                    grouped_shards_to_dispatch[target_client_id] = shard_list_for_target
                else:
                    logger.warning(f"Could not find client_id for target_rank {target_rank} during dispatch preparation.")

        # 4. 清空累加器，释放内存
        self.remote_grad_accumulator.clear()
        
        return grouped_shards_to_dispatch

    def add_received_grad_shard(self, content: dict):
        """
        由 Trainer 调用，将从网络接收到的梯度分片添加到缓冲区。
        """
        param_name = content.get('param_name')
        grad_shard_data = content.get('grad_shard_data')
        received_step = content.get('step')

        if param_name and grad_shard_data is not None and received_step is not None:
            # --- 核心修正：确保添加到 buffer 的是 Tensor ---
            if isinstance(grad_shard_data, torch.Tensor):
                if param_name in self.grad_shard_buffer:
                    self.grad_shard_buffer[param_name].append((grad_shard_data.to(self.device), received_step))
                    logger.debug(f"Rank {self.rank}: Received and buffered grad shard for '{param_name}' from step {received_step}.")
                else:
                    logger.warning(f"Rank {self.rank}: Received grad shard for unmanaged param '{param_name}'.")
            else:
                logger.error(f"FATAL ERROR in add_received_grad_shard: grad_shard_data for '{param_name}' is not a Tensor, but {type(grad_shard_data)}. Data will be dropped.")
        else:
            logger.warning(f"Received incomplete grad shard content: {content.keys()}")

    def update_parameter_shards(self, shard_list: list):
        """
        由 Trainer 调用，用接收到的参数分片更新本地的完整模型。
        """
        for content in shard_list:
            param_name = content.get('param_name')
            shard_data = content.get('shard_data')
            source_rank = content.get('source_rank')

            if not all([param_name, shard_data is not None, source_rank is not None]):
                logger.warning("Received incomplete parameter shard update. Ignoring.")
                return

            sharding_info = self.sharding_metadata.get(param_name)
            if sharding_info:
                self.reconstructor.update_cache(param_name, shard_data, sharding_info, source_rank)
                logger.debug(f"Rank {self.rank}: Updated cache for '{param_name}' with shard from rank {source_rank}.")

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
        if self.aggregation_mode in ['model', 'local-reconstruct', 'fl-sim']:
            if not self.local_optimizer:
                logger.warning("local_optimizer not initialized. Returning empty tensor.")
                return torch.tensor([])
            params_to_vectorize = self.local_optimizer.param_groups[0]['params']
            return parameters_to_vector([p.data for p in params_to_vectorize]).clone()

        elif self.aggregation_mode == 'shardedGradient':            
            if not self.reconstructor:
                logger.error("Reconstructor not available. Cannot get full params.")
                return torch.tensor([])

            reconstructed_params_list = []
            with torch.no_grad():
                for name in self.ordered_trainable_names:
                    full_param = None
                    
                    # # --- 优先尝试从 GPU 缓存获取 ---
                    if name in self.reconstructor.gpu_cache:
                        # 从 GPU 缓存获取，需要 clone 以免修改缓存中的原始数据
                        full_param = self.reconstructor.gpu_cache[name].clone()
                        logger.debug(f"Getting param '{name}' from GPU cache.")
                    # --- 回退到 CPU 缓存 ---
                    elif name in self.reconstructor.cpu_cache:
                        # 从 CPU 缓存获取，需要 clone
                        full_param = self.reconstructor.cpu_cache[name].clone()
                        logger.debug(f"Getting param '{name}' from CPU cache.")
                    else:
                        logger.warning(f"Parameter '{name}' not found in any cache. Skipping.")
                        continue
                    
                    # 2. 用自己最新的持久化分片去“修正”它
                    if name in self._my_persistent_shards:
                        my_latest_shard_param = self._my_persistent_shards[name]
                        
                        sharding_info = self.sharding_metadata[name]
                        split_dim = sharding_info['split_dim']
                        my_rank_info = sharding_info['ranks_info'][self.rank]
                        offset, size = my_rank_info['offset'], my_rank_info['size']
                        
                        # 在克隆出的完整参数上进行就地修正
                        # 确保分片和目标张量在同一设备上
                        target_device = full_param.device
                        shard_data_to_copy = my_latest_shard_param.data.to(target_device)
                        full_param.narrow(split_dim, offset, size).copy_(shard_data_to_copy)

                    # 3. 将最终重组好的【CPU】张量加入列表
                    #    为了后续的 parameters_to_vector 效率和一致性，最好都在 CPU 上
                    reconstructed_params_list.append(full_param.cpu())

            if not reconstructed_params_list:
                logger.warning("Reconstructed params list for vectorization is empty.")
                return torch.tensor([])
            
            # 4. 将重组后的完整参数列表转换为向量
            return parameters_to_vector(reconstructed_params_list)
        
        else:
            logger.warning(f"get_full_trainable_params_as_vector not implemented for mode {self.aggregation_mode}")
            return torch.tensor([])
        
    def set_full_trainable_params_from_vector(self, full_vector: torch.Tensor):
        """
        从一个完整的 FP32 向量中，恢复所有可训练参数的状态。
        """
        if not self.local_optimizer:
            logger.warning("`set_full_trainable_params_from_vector` called but local_optimizer is not initialized.")
            return
        
        if self.aggregation_mode in ['model', 'local-reconstruct', 'fl-sim']:
            if not self.local_optimizer: return
            params_to_update = self.local_optimizer.param_groups[0]['params']
            vector_to_parameters(full_vector.to(params_to_update[0].device), params_to_update)
            # (你可能还需要一个写回到低精度模型参数的步骤)
            
        elif self.aggregation_mode == 'shardedGradient':
            logger.info(f"Rank {self.rank}: Setting full trainable params from vector...")
            
            # 1. 先将完整的向量，按照模型参数的顺序，拆分成【完整】的参数张量列表
            #    我们需要一个模板来获取形状
            template_params = [self.name_to_param_map[name] for name in self.ordered_trainable_names]
            
            # 确保 full_vector 在正确的设备上
            full_vector = full_vector.to(self.device)
            
            # 使用 vector_to_parameters 将向量写回到一个【临时】的参数列表
            # 这里有一个技巧：我们可以直接写回到模型参数中，然后再切分
            # 但是更安全的方法是创建一个临时列表
            temp_full_params = [torch.empty_like(p) for p in template_params]
            vector_to_parameters(full_vector, temp_full_params)

            # 2. 遍历这些临时的完整参数，切出属于我的分片，并更新我的持久化分片
            with torch.no_grad():
                for i, name in enumerate(self.ordered_trainable_names):
                    if name in self._my_persistent_shards:
                        full_param_restored = temp_full_params[i]
                        
                        sharding_info = self.sharding_metadata[name]
                        split_dim = sharding_info['split_dim']
                        my_rank_info = sharding_info['ranks_info'][self.rank]
                        offset, size = my_rank_info['offset'], my_rank_info['size']
                        
                        # 从恢复的完整参数中切出我的分片
                        shard_to_set = full_param_restored.narrow(split_dim, offset, size)
                        
                        # 更新我负责的持久化分片
                        persistent_shard_param = self._my_persistent_shards[name]
                        persistent_shard_param.data.copy_(shard_to_set)

            logger.info(f"Rank {self.rank}: Persistent shards have been updated from the provided vector.")

    def _calculate_adaptive_alpha(self, received_step):
        """一个辅助函数，用于计算自适应的聚合权重。"""
        initial_alpha = 0.5
        staleness = (self.current_step - received_step) / self.aggregation_steps
        logger.debug(f"staleness: {staleness}")
        # 如果消息来自未来，则不认为是陈旧的
        if staleness < 0: 
            return initial_alpha
            
        func_type = self.adaptive_weight_cfg.get('type', 'constant')
        if func_type == 'exponential':
            decay_rate = self.adaptive_weight_cfg.get('decay_rate', 0.9)
            logger.debug(
            f"[AdaptiveWeight] current_step={self.current_step}, received_step={received_step}, "
            f"calculated_alpha={initial_alpha * (decay_rate ** staleness):.4f}"
        )
            return initial_alpha * (decay_rate ** staleness)
        elif func_type == 'linear':
            decay_factor = self.adaptive_weight_cfg.get('decay_factor', 0.01)
            return max(0.0, initial_alpha - staleness * decay_factor)
        # ... (可以扩展其他函数)
        else: # 'constant'
            return initial_alpha
        
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
        for name, module in self.model.named_modules():
             # name 是全局唯一的名称，例如 'transformer.h.0.attn'
             if name: # 跳过根模块
                 module.fedspeed_name = name
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
        """
        核心方法：执行分片和演练，返回所有客户端的初始化数据包。
        根据 config.federate.aggregation_mode 采取不同的打包策略。
        """
        logger.info(f"[Coordinator] Preparing metadata packages for world_size={self.world_size}...")

        # --- 1. 获取通用配置 ---
        aggregation_mode = self.cfg.federate.get('aggregation_mode', 'gradient')
        noniid = self.cfg.federate.two_stages.get('noniid', False)

        # --- 2. 计算所有客户端共享的元数据 ---
        
        # a. 引用计数 (只在 'gradient' 模式下需要)
        ref_counts_package = self._profile_ref_counts() if aggregation_mode == 'gradient' else None

        # b. 分组元数据 (如果启用)
        grouping_metadata = self._calculate_grouping_and_layer_assignments() if noniid else {}

        # c. 分片元数据 (只在 'shardedGradient' 和 'gradient' 模式下需要)
        sharding_metadata = OrderedDict()
        if aggregation_mode in ['shardedGradient', 'gradient']:
            logger.info(f"[Coordinator] Mode: '{aggregation_mode}'. Preparing sharding metadata.")
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    # 只为可训练参数或冻结的分片参数计算元数据
                    is_trainable = param.requires_grad
                    # 在 'gradient' 模式下，所有参数都分片；
                    # 在 'shardedGradient' 模式下，只有可训练参数分片
                    should_shard = (aggregation_mode == 'gradient' or 
                                    (aggregation_mode == 'shardedGradient' and is_trainable))
                                    
                    if not should_shard:
                        continue

                    dim_to_split = 0
                    dim_size = param.shape[dim_to_split]

                    if dim_size < self.world_size:
                        split_sizes = [dim_size] + [0] * (self.world_size - 1)
                    else:
                        split_sizes = self._get_even_split_sizes(dim_size)
                    
                    offsets = [0] + list(np.cumsum(split_sizes)[:-1])
                    
                    sharding_metadata[name] = {
                        'split_dim': dim_to_split,
                        'ranks_info': {
                            rank: {'size': int(size), 'offset': int(offset)}
                            for rank, (size, offset) in enumerate(zip(split_sizes, offsets))
                        }
                    }

        # --- 3. 为所有客户端构建一个【统一的、轻量级的】元数据包 ---
        base_package_content = {
            'ref_counts': ref_counts_package,
            'sharding_metadata': sharding_metadata,
            # full_model_state_dict 和 shards 字段被彻底移除
        }
        if noniid:
            base_package_content.update(grouping_metadata)
            
        # --- 4. 为每个客户端创建包 (内容完全相同) ---
        all_client_packages = [base_package_content.copy() for _ in range(self.world_size)]
        
        logger.info("[Coordinator] All lightweight metadata packages created successfully.")
        return all_client_packages

    def _calculate_grouping_and_layer_assignments(self):
        """
        【新函数】计算客户端分组、层分配和跨组通信映射。
        """
        num_groups = self.cfg.federate.two_stages.get('group_num', 2)
        if self.world_size < num_groups:
            raise ValueError(f"world_size ({self.world_size}) cannot be smaller than num_groups ({num_groups}).")

        client_ranks = list(range(self.world_size))

        # --- 1. 健壮的客户端分组 (支持不均匀) ---
        # 使用 torch.chunk 来获得最均衡的分组
        group_assignments = torch.chunk(torch.tensor(client_ranks), num_groups)
        group_info = {i: group.tolist() for i, group in enumerate(group_assignments)}
        
        client_to_group_map = {}
        for group_id, members in group_info.items():
            for rank in members:
                client_to_group_map[rank] = group_id
        
        logger.info(f"Grouping results ({num_groups} groups): {group_info}")
        
        # --- 2. 识别模型中的逻辑层 (不变) ---
        block_class_name = self.cfg.federate.get('transformer_block_class_name')
        if not block_class_name:
            raise ValueError("`transformer_block_class_name` must be specified for grouping.")
        
        logical_layers = []
        for module in self.model.modules():
            if module.__class__.__name__ == block_class_name:
                if not hasattr(module, 'fedspeed_name'):
                    raise AttributeError(f"Module {module.__class__.__name__} is missing 'fedspeed_name'.")
                logical_layers.append(module)
        
        total_layers = len(logical_layers)
        if total_layers == 0:
            raise ValueError(f"No logical layers of type '{block_class_name}' found in the model.")

        # --- 3. 【核心修改】在每个组内部独立进行层分配 ---
        client_to_layers_map = {}
        # 这个字典用于构建通信图: slot_idx -> [ranks]
        # slot_idx 代表一个组内成员的索引 (第0个成员, 第1个成员...)
        peers_by_slot = {}

        for group_id, members in group_info.items():
            num_members = len(members)
            if total_layers < num_members:
                 raise ValueError(f"In group {group_id}, total layers ({total_layers}) is less than group size ({num_members}). Cannot assign layers.")
            
            layer_indices = torch.arange(total_layers)
            # 在组内部分配层
            layer_chunks_in_group = torch.chunk(layer_indices, num_members)
            
            for i, member_rank in enumerate(members):
                # a. 分配层
                assigned_layers_indices = layer_chunks_in_group[i].tolist()
                client_to_layers_map[member_rank] = assigned_layers_indices
                
                # b. 填充 peers_by_slot，用于下一步构建通信图
                slot_idx = i # 组内索引
                if slot_idx not in peers_by_slot:
                    peers_by_slot[slot_idx] = []
                peers_by_slot[slot_idx].append(member_rank)

        # --- 4. 构建跨组通信映射 ---
        layer_to_peers_map = {i: [] for i in range(total_layers)}
        for layer_idx in range(total_layers):
            # 对于每一层，遍历所有客户端，看谁负责它
            for rank in client_ranks:
                if layer_idx in client_to_layers_map.get(rank, []):
                    layer_to_peers_map[layer_idx].append(rank)

        logger.debug(f"Client layer assignments (in-group model parallelism): {client_to_layers_map}")
        logger.debug(f"Layer-to-peer communication map: {layer_to_peers_map}")

        logical_layers_names = [m.fedspeed_name for m in logical_layers]

        return {
            "grouping_metadata": {
                "client_to_group_map": client_to_group_map,
                "group_info": group_info,
                "client_to_layers_map": client_to_layers_map,
                "layer_to_peers_map": layer_to_peers_map,
                "logical_layers_names": logical_layers_names 
            }
        }

    def _profile_ref_counts(self):
        """
        辅助方法：执行一次“演练”来计算所有参数的引用计数。
        (这个方法是从你之前的代码中提取和重构的，逻辑保持不变)
        """
        logger.info("[Coordinator] Starting reference count profiling...")
        
        # 准备一个假的输入
        dummy_input_ids = torch.ones(1, self.cfg.llm.tok_len, dtype=torch.long, device=self.device)
        dummy_labels = torch.ones(1, self.cfg.llm.tok_len, dtype=torch.long, device=self.device)
        dummy_input = {'input_ids': dummy_input_ids, 'labels': dummy_labels}

        # 应用激活检查点 (如果配置了)
        if self.cfg.federate.use_activation_checkpointing:
            # 1. 从配置中动态获取块类名的字符串
            block_class_name = self.cfg.federate.get('transformer_block_class_name')
            if not block_class_name:
                logger.warning(
                    "`use_activation_checkpointing` is True, but `transformer_block_class_name` is not specified. "
                    "Profiling will proceed without checkpointing, which may lead to inaccurate reference counts."
                )
            else:
                # 2. 在模型中找到这个类对象
                block_class = None
                # CoordinatorEngine 的 self.model 是原始模型，可以直接搜索
                for module in self.model.modules():
                    if module.__class__.__name__ == block_class_name:
                        block_class = module.__class__
                        break
                
                if block_class:
                    # 3. 使用找到的类对象来应用检查点
                    logger.info(f"[Coordinator] Applying activation checkpointing for profiling to block class: {block_class.__name__}")
                    self._apply_checkpointing_for_profile(block_class)
                else:
                    available_classes = {m.__class__.__name__ for m in self.model.modules()}
                    logger.warning(
                        f"Could not find the specified block class '{block_class_name}' in the model for profiling. "
                        f"Proceeding without checkpointing. Available module classes include: {list(available_classes)[:10]}..."
                    )

        # 注册演练钩子
        fwd_counter, recompute_counter, bwd_counter = {}, {}, {}
        
        def post_fwd_profiling_hook(module, inputs, outputs):
            counter = recompute_counter if self.is_recomputing else fwd_counter
            param_ids = self.module_to_params_map.get(module.fedspeed_id, [])
            for pid in param_ids:
                counter[pid] = counter.get(pid, 0) + 1

        fwd_handle = self.model.register_forward_hook(post_fwd_profiling_hook)
        bwd_handle = self.model.register_full_backward_hook(self._profiling_hook_factory(bwd_counter))

        # 执行演练
        self.model.train()
        try:
            outputs = self.model(**dummy_input)
            loss = outputs.loss
            loss.backward()
        except Exception as e:
            logger.error(f"[Coordinator] Profiling failed with error: {e}. Ref counts will be empty.")
        finally:
            # 清理钩子
            fwd_handle.remove()
            bwd_handle.remove()

        logger.info("[Coordinator] Profiling complete.")

        return {
            'fwd': fwd_counter,
            'recompute': recompute_counter,
            'bwd': bwd_counter,
            'id_to_param_map_keys': list(self.id_to_param_map.keys())
        }

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

def _sanitize_model_name_for_path(model_name: str) -> str:
    base_name = os.path.basename(model_name)
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', base_name)

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