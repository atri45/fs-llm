import logging
import pickle
import base64
import time
import torch
import torch.distributed as dist
import random
import os
import queue
import psutil
import numpy as np

from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.trainers.context import CtxVar
from torch.utils.checkpoint import checkpoint
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.llm.trainer.fedspeed_engine import FedSpeedEngine, log_memory_usage
from federatedscope.register import register_trainer
from federatedscope.core.message import Message
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.llm.trainer.anonymous_router import AnonymousRouter
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class EarlyStopper:
    def __init__(self, patience: int, delta: float = 0, checkpoint_path: str = "best_model.ckpt"):
        """
        Args:
            patience (int): 在性能没有提升的情况下，要等待多少次评估。
            delta (float): 认为性能“提升”所需的最小变化量。
            checkpoint_path (str): 保存最佳模型状态的临时文件路径。
        """
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float, engine: FedSpeedEngine):
        """
        在每次评估后调用此方法，检查本地早停条件。
        """
        is_invalid_score = current_score is None or np.isnan(current_score) or np.isinf(current_score)

        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(engine)
            return
        
        if is_invalid_score or current_score > self.best_score + self.delta:
            # 性能变差了
            self.counter += 1
            status = "invalid (NaN/Inf)" if is_invalid_score else "no improvement"
            logger.info(f"[EarlyStopper] Score is {status}. Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 性能提升了
            self.best_score = current_score
            self.save_checkpoint(engine)
            if not self.early_stop:
                self.counter = 0

    def check_global_consensus_and_stop(self, global_votes: dict):
        """
        检查全局投票共识，如果达成，则抛出异常。
        """
        if not global_votes:
            return
            
        all_voted = all(global_votes.values())
        if all_voted:
            logger.info(f"--- Global early stop consensus reached! Terminating training. Vote status: {global_votes} ---")
            raise EarlyStopException() # 抛出异常以终止训练

    def save_checkpoint(self, engine: FedSpeedEngine):
        """保存当前最佳的模型参数。"""
        logger.info(f"[EarlyStopper] New best score: {self.best_score:.4f}. Saving model state...")
        best_params_vec = engine.get_full_trainable_params_as_vector()
        torch.save(best_params_vec, self.checkpoint_path)

    def load_best_checkpoint(self, engine: FedSpeedEngine):
        """将模型恢复到最佳状态。"""
        if os.path.exists(self.checkpoint_path):
            best_params_vec = torch.load(self.checkpoint_path, map_location=engine.device)
            engine.set_full_trainable_params_from_vector(best_params_vec)
            logger.info(f"Rank {engine.rank}: Successfully loaded best model state. Best score: {self.best_score:.4f}.")


class EarlyStopException(Exception):
    pass


class PerformanceMonitor:
    def __init__(self, device):
        self.device = device
        self.start_time = 0.0
        self.end_time = 0.0
        self.self_training_seconds = 0.0
        self.total_training_seconds = 0.0 # 用于累加纯训练时间
        self._last_resume_time = 0.0
        self._is_running = False
        self._is_paused = False

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
        if self._is_running:
            return # 防止重复启动
        # 使用辅助函数进行判断
        if self._is_cuda():
            torch.cuda.reset_peak_memory_stats(self.device)
        
        self.start_time = time.time()
        self._last_resume_time = self.start_time
        self._is_running = True
        self._is_paused = False
        logger.info("Performance monitor started.")
        
    def pause(self):
        """暂停计时，将当前时间段累加到总时间中。"""
        if self._is_running and not self._is_paused:
            current_time = time.time()
            self.total_training_seconds += (current_time - self._last_resume_time)
            self._is_paused = True
            logger.debug("Performance monitor paused.")

    def resume(self):
        """恢复计时。"""
        if self._is_running and self._is_paused:
            self._last_resume_time = time.time()
            self._is_paused = False
            logger.debug("Performance monitor resumed.")
    
    def self_stop(self):
        """记录本地收敛所用训练时间。"""
        if not self._is_running:
            return

        self.self_training_seconds = self.total_training_seconds
        
    def stop(self):
        """停止计时，收集峰值数据，并打印报告。"""
        if not self._is_running:
            return

        self.end_time = time.time()
        
        if not self._is_paused:
            self.total_training_seconds += (self.end_time - self._last_resume_time)

        # 收集峰值数据
        self.cpu_mem_usage_peak = self.process.memory_info().rss / (1024 ** 2)
        
        if self._is_cuda():
            stats = torch.cuda.memory_stats(self.device)
            self.gpu_mem_allocated_peak = stats["allocated_bytes.all.peak"] / (1024 ** 2)
            self.gpu_mem_reserved_peak = stats["reserved_bytes.all.peak"] / (1024 ** 2)

        self.report()
        self._is_running = False

    def report(self):
        """打印性能报告。"""
        total_wall_clock_seconds = self.end_time - self.start_time
        total_wall_clock_hours = total_wall_clock_seconds / 3600.0
        
        # 纯训练时间
        self_training_hours = self.self_training_seconds / 3600.0
        total_training_hours = self.total_training_seconds / 3600.0

        logger.info("----------- Performance Report -----------")
        logger.info(f"  - Total Wall-Clock Time: {total_wall_clock_hours:.2f} hours ({total_wall_clock_seconds:.2f} seconds)")
        logger.info(f"  - Pure Self Training Time (excluding evaluation): {self_training_hours:.2f} hours ({self.self_training_seconds:.2f} seconds)")
        logger.info(f"  - Pure Total Training Time (excluding evaluation): {total_training_hours:.2f} hours ({self.total_training_seconds:.2f} seconds)")
        
        logger.info(f"  - CPU Memory Peak Usage (RSS): {self.cpu_mem_usage_peak / 1024.0:.2f} GB ({self.cpu_mem_usage_peak} MB)")
        
        if self._is_cuda():
            logger.info(f"  - GPU Memory Peak Allocated: {self.gpu_mem_allocated_peak / 1024.0:.2f} GB ({self.gpu_mem_allocated_peak} MB)")
            logger.info(f"  - GPU Memory Peak Reserved: {self.gpu_mem_reserved_peak / 1024.0:.2f} GB ({self.gpu_mem_reserved_peak} MB)")
        else:
            logger.info("  - GPU Monitoring: Not available (CUDA not found or not used).")
        logger.info("------------------------------------------")


class StepTimer:
    """
    一个通用的上下文管理器，用于测量代码块的执行时间并进行累加。
    """
    def __init__(self, name: str):
        self.name = name
        self._total_seconds = 0.0
        self._start_time = 0.0
        self._is_running = False

    def start(self):
        if not self._is_running:
            self._start_time = time.perf_counter()
            self._is_running = True

    def stop(self):
        if self._is_running:
            end_time = time.perf_counter()
            self._total_seconds += (end_time - self._start_time)
            self._is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def total_time(self):
        return self._total_seconds

    def report(self):
        logger.info(f"  - Total Time in {self.name}: {self.total_time:.4f} seconds")

# --- 现在定义具体的计时器 ---
class CommunicationTimer(StepTimer):
    def __init__(self):
        super().__init__("Communication")

class ForwardTimer(StepTimer):
    def __init__(self):
        super().__init__("Forward")

class BackwardTimer(StepTimer):
    def __init__(self):
        super().__init__("Backward")

class UpdateTimer(StepTimer):
    def __init__(self):
        super().__init__("Update")

class AggregationTimer(StepTimer):
    def __init__(self):
        super().__init__("Aggregation")


class FedSpeedTrainer(LLMTrainer):
    def __init__(self, model, data, device, config, **kwargs):
        super().__init__(model, data, device, config, **kwargs)
        self.config = config
        self.fedspeed_engine = None
        self.fedspeed_init_package = None
        self.current_step = 0
        self.comm_manager = None
        self.client_id = -1
        self.peers = []
        self.num_batches = 0
        self.router = None
        self.internal_message_queue = queue.Queue()
        self.comm_executor = None
        self.monitor = PerformanceMonitor(device)
        # 初始化所有计时器
        self.communication_timer = CommunicationTimer()
        self.forward_timer = ForwardTimer()
        self.backward_timer = BackwardTimer()
        self.update_timer = UpdateTimer()
        self.aggregation_timer = AggregationTimer()

        self.aggregation_mode = self.config.federate.get('aggregation_mode') 
        self.is_decentralized_sharded_mode = (self.aggregation_mode == 'shardedGradient')
        self.aggregation_steps = self.config.federate.get('aggregation_steps') 
        self.stage_one_steps = self.cfg.federate.two_stages.get('stage_one_steps', 250)
        self.asynchronous = self._cfg.federate.get('asynchronous_aggregation', False)
        self.adaptive_weight_cfg = self.cfg.federate.get('adaptive_weight', {'use': False})
        self.gossip_num = self._cfg.federate.get('gossip_num', 1)
        self.eval_freq = self._cfg.eval.get('freq')
        self.anonymous_routing = self.cfg.federate.get('anonymous_routing', False)
        if self.aggregation_mode in ['fl-sim'] :
            self.asynchronous = False

        # self.comm_executor = ThreadPoolExecutor(max_workers=4)
        if self.cfg.early_stop.patience > 0:
            self.early_stopper = EarlyStopper(
                patience=self.cfg.early_stop.patience,
                checkpoint_path=os.path.join(self.cfg.outdir, "fedspeed_best_model.ckpt")
            )
            # 记录每个 client_id 的早停投票状态 (True/False)
            self.global_early_stop_votes = {}
        else:
            self.early_stopper = None

    def _run_batch(self, hooks_set, run_step=-1):
        """
        重写父类的 _run_batch 方法，以支持在跳过 NaN batch 时，
        不计入总 step 数，并从 data_loader 中获取新的 batch 来补偿。
        """
        if run_step == -1:
            run_step = getattr(self.ctx, f"num_{self.ctx.cur_split}_batch")

        # 确保 dataloader 是可重新迭代的
        if not isinstance(self.ctx.get(f'{self.ctx.cur_split}_loader'), ReIterator):
             self.ctx[f'{self.ctx.cur_split}_loader'] = ReIterator(self.ctx.get(f'{self.ctx.cur_split}_loader'))
        
        # 使用 while 循环，由我们自己的有效 step 计数器 self.current_step 控制
        while self.current_step < run_step:
            try:
                # 在 on_batch_start 之前获取数据
                for hook in hooks_set["on_batch_start"]:
                    hook(self.ctx)

                for hook in hooks_set["on_batch_forward"]:
                    hook(self.ctx)

                # 检查在 on_batch_forward 中是否设置了跳过标志
                if getattr(self.ctx, 'skip_this_batch', False):
                    logger.warning(
                        f"[Step Drift Prevented] Batch resulted in NaN. "
                        f"Skipping backward/end hooks and fetching a new batch. "
                        f"Effective step count remains {self.current_step}."
                    )
                    # 直接进入下一次 while 循环，获取新数据
                    # self.current_step 不会增加
                    continue 

                for hook in hooks_set["on_batch_backward"]:
                    hook(self.ctx)

                for hook in hooks_set["on_batch_end"]:
                    hook(self.ctx)

            except StopIteration:
                # 数据加载器已经耗尽
                logger.warning("Dataloader exhausted before reaching target steps. "
                               f"Completed {self.current_step}/{run_step} effective steps.")
                break # 退出 while 循环

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        # a. 获取总的目标 step 数
        total_steps = self.cfg.train.local_update_steps
        
        # b. 将 epoch 数强制设为 1
        ctx.num_train_epoch = 1
        
        # c. 将一个 "epoch" 内的 batch 数，直接设置为总的 step 数
        ctx.num_train_batch = total_steps
        ctx.num_train_batch_last_epoch = total_steps # 也更新这个

        logger.info(
            f"Step-based training is enabled. Overriding training loop length: "
            f"num_train_epoch=1, num_train_batch={total_steps}."
        )
        
        # d. 确保 dataloader 被 ReIterator 包裹，以支持超过一轮的数据迭代
        loader_key = "train_loader"
        dataloader = ctx.get(loader_key)
        if dataloader is not None and not isinstance(dataloader, ReIterator):
            ctx[loader_key] = ReIterator(dataloader)
            logger.info(f"Wrapped '{loader_key}' with ReIterator for step-based training.")
            
        # 检查并创建FedSpeedEngine
        if self.fedspeed_engine is None and dist.is_initialized():
            if self.fedspeed_init_package is None:
                raise RuntimeError("FedSpeed Trainer started but init package was not set by the client.")

            init_package = self.fedspeed_init_package
            self.aggregation_mode = self.config.federate.get('aggregation_mode') 
            self.fedspeed_engine = FedSpeedEngine(
                ctx.model, 
                self.cfg, 
                ctx.device, 
                dist.group.WORLD,
                initial_shards=init_package,
                ref_counts=init_package['ref_counts']
            )
            # 激活检查点
            if self.cfg.federate.use_activation_checkpointing:
            # 1. 从配置中动态获取块类名的字符串
                block_class_name = self.cfg.federate.get('transformer_block_class_name')
                if not block_class_name:
                    raise ValueError(
                        "`federate.use_activation_checkpointing` is True, but "
                        "`federate.transformer_block_class_name` is not specified in the config."
                    )

                # 2. 在加载的模型中找到这个类对象
                block_class = None
                # 我们需要遍历内部的 PeftModel 或 base_model
                model_to_search = ctx.model
                if hasattr(model_to_search, 'model'): # AdapterModel -> PeftModel
                    model_to_search = model_to_search.model
                    
                for module in model_to_search.modules():
                    if module.__class__.__name__ == block_class_name:
                        block_class = module.__class__
                        break
                
                if block_class is None:
                    available_classes = {m.__class__.__name__ for m in model_to_search.modules()}
                    raise ValueError(
                        f"Could not find the specified block class '{block_class_name}' in the model for activation checkpointing. "
                        f"Available module classes include (first 10): {list(available_classes)[:10]}..."
                    )
                
                # 3. 使用找到的类对象来应用检查点
                logger.info(f"Applying activation checkpointing to block class: {block_class.__name__}")
                apply_checkpointing(ctx.model, block_class, self.fedspeed_engine)

            self.fedspeed_engine._register_hooks_recursively(self.fedspeed_engine.model)
            
            # 初始化投票状态表
            if self.cfg.early_stop.patience > 0:
                all_client_ids = list(init_package['rank_to_client_id_map'].values())
                self.global_early_stop_votes = {cid: False for cid in all_client_ids}
                self.global_early_stop_votes[self.client_id] = False

            # 清理，防止内存泄漏
            self.fedspeed_init_package = None

            # 在所有训练开始前，启动 Reconstructor 的后台线程
            if self.fedspeed_engine and self.fedspeed_engine.reconstructor:
                self.fedspeed_engine.reconstructor.start()
            self.fedspeed_engine.aggregation_timer = self.aggregation_timer
            self.monitor.start()

            if self.fedspeed_engine.is_hybrid_mode:
                self.is_decentralized_sharded_mode = False
                self.aggregation_mode = 'local-reconstruct'

    # 将所有复杂逻辑都委托给 Engine
    def _hook_on_batch_forward(self, ctx):
        logger.debug(f"batch_forward")
        # 调用引擎的forward方法
        #    所有的All-Gather和释放都会在内部被钩子自动处理
        data_batch_on_device = {k: v.to(ctx.device) for k, v in ctx.data_batch.items()}
                # 检查 attention_mask 的类型，并进行转换以兼容 adapters 库
        if 'attention_mask' in data_batch_on_device and data_batch_on_device['attention_mask'].dtype == torch.bool:
            
            # 'adapters' 库的内部实现需要数值类型的 attention_mask
            # 我们将其转换为 long 类型 (0 和 1)
            logger.debug("Converting boolean attention_mask to long for adapter compatibility.")
            data_batch_on_device['attention_mask'] = data_batch_on_device['attention_mask'].long()
        log_memory_usage("Forward Start")
        self.forward_timer.start()
        outputs = self.fedspeed_engine.forward(**data_batch_on_device)
        self.forward_timer.stop()
        log_memory_usage("Forward End")
        
        # 将结果保存到 ctx 中
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # 检查 loss 是否为 NaN
            if torch.isnan(outputs.loss):
                ctx.skip_this_batch = CtxVar(True, "batch")
                logger.warning(
                    f"[Step {self.current_step}] Loss is NaN. Skipping this batch. "
                )
                # 即使跳过，也要为下游钩子提供一个默认的零值loss
                ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device), "batch")
                ctx.loss_task = CtxVar(torch.tensor(0.0, device=ctx.device), "batch")
            else:
                ctx.skip_this_batch = CtxVar(False, "batch")
                ctx.loss_batch = CtxVar(outputs.loss, "batch")
                ctx.loss_task = CtxVar(outputs.loss, "batch") # 简化
        else:
            # 处理推理或未提供label的情况
            ctx.skip_this_batch = CtxVar(True, "batch")           
            ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device), "batch")
            ctx.loss_task = CtxVar(torch.tensor(0.0, device=ctx.device), "batch")
            if ctx.cur_mode == MODE.TRAIN: # 只在训练模式下打印警告
                 logger.warning("No loss returned from model forward pass. Skipping backward.")
        
        # ... 其他 ctx 变量的赋值，例如 logits
        if hasattr(outputs, 'logits'):
            ctx.y_prob = CtxVar(outputs.logits, "batch")
        if 'labels' in ctx.data_batch:
            ctx.y_true = CtxVar(ctx.data_batch['labels'], "batch")
        ctx.batch_size = CtxVar(len(ctx.data_batch['input_ids']), "batch")

    def _hook_on_batch_backward(self, ctx):
        logger.debug(f"batch_backward")
        if ctx.skip_this_batch:
            return
        if hasattr(ctx, 'loss_task') and ctx.loss_task.requires_grad:
            # 这一步只负责调用 engine.backward()，它会完成流式的参数管理和梯度收集
            log_memory_usage("Backward Start")
            self.backward_timer.start()
            self.fedspeed_engine.backward(ctx.loss_task)
            self.backward_timer.stop()
            log_memory_usage("Backward End (Grads Collected)")

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            # 在跳过时，也要确保清理可能存在的状态
            if hasattr(self.fedspeed_engine, 'temp_full_gradients'):
                self.fedspeed_engine.temp_full_gradients.clear()
            return
        self.current_step += 1

        # --- 检查是否需要阶段转换 ---
        if self.fedspeed_engine.is_hybrid_mode and self.current_step == self.stage_one_steps:
            self.is_decentralized_sharded_mode = True
            self.aggregation_mode = 'shardedGradient'              
            self.fedspeed_engine.aggregation_mode = 'shardedGradient'
            self.fedspeed_engine.local_optimizer = None
            self.fedspeed_engine._initialize_with_reconstructor()
            self.fedspeed_engine._initialize_for_shardedGradient()
        if self.current_step == self.stage_one_steps:
            if self.fedspeed_engine.noniid:
                self._trigger_stage_transition()
            # else:
            #     self.fedspeed_engine.sync_model_layers_via_broadcast()
            #     logger.info("--- Layer-wise sync complete. Now entering Stage 2. ---")

        # 通用异步逻辑 (消息处理)
        if self.fedspeed_engine.noniid or not self.fedspeed_engine.is_in_stage_one:
            logger.debug(f"process_incoming_messages")
            self.aggregation_timer.start()
            self.process_incoming_messages()   
            self.aggregation_timer.stop() 

        # 梯度分片处理流程
        if self.is_decentralized_sharded_mode:
            self.fedspeed_engine.current_step = self.current_step
            # 1. 处理本地计算的梯度：保留自己的分片，准备并匿名发送其他人的分片
            logger.debug(f"process_and_dispatch_gradients")
            my_local_grad_shards, grouped_shards_to_dispatch = self.fedspeed_engine.process_and_dispatch_gradients()

            # 2. 匿名发送梯度分片
            if grouped_shards_to_dispatch:
                logger.debug(f"send_gradient_shard_anonymously")
                self.aggregation_timer.start()
                self.communication_timer.start()
                for target_client_id, shard_list in grouped_shards_to_dispatch.items():
                    if shard_list: # 确保列表不为空
                        self.send_gradient_shards_anonymously_packaged(
                            shard_list, # 发送整个列表
                            target_client_id
                        )
                self.communication_timer.stop()
                self.aggregation_timer.stop()

            # 3. 执行优化器步骤 (内部包含梯度 FedAvg 聚合)
            logger.debug(f"sharded_step")
            self.update_timer.start()
            updated_shards_for_broadcast, model_was_updated_by_peers = self.fedspeed_engine.sharded_step(my_local_grad_shards)
            self.update_timer.stop()

            # 4. 异步推送更新后的参数分片 (非匿名)
            if model_was_updated_by_peers and not self.fedspeed_engine.is_in_stage_one:
                logger.debug(f"broadcast_shard_updates")
                self.aggregation_timer.start()
                self.communication_timer.start()
                self.broadcast_shard_updates(updated_shards_for_broadcast)
                self.communication_timer.stop()
                self.aggregation_timer.stop()

        else:
            # 常规流程
            # 1. 聚合梯度
            log_memory_usage("Aggregate Grads Start")            
            aggregated_grads_dict = self.fedspeed_engine.aggregate_gradients()
            log_memory_usage("Aggregate Grads End (Full Grads Cleared)")

            # 2. 执行优化器步骤
            log_memory_usage("Optimizer Step Start")
            params_to_send = self.fedspeed_engine.step(aggregated_grads_dict)
            log_memory_usage("Optimizer Step End")

            # 3. 通信
            if params_to_send is not None:
                self.aggregation_timer.start()
                self._send_data(params_to_send)
                
                # 4. 等待并处理同步消息
                if self.aggregation_mode == 'fl-sim':
                    message = self.comm_manager.receive()
                    if message.msg_type == 'aggregated_model_para':
                        self.handle_model(message)

                self.aggregation_timer.stop()

        # 回收显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 日志和评估
        my_rank = dist.get_rank()
        logger.info(f"--- Rank {my_rank} | Step {self.current_step} | Loss: {ctx.loss_batch.item():.4f} ---")
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        if self.eval_freq > 0 and (self.current_step % self.eval_freq) == 0 and not self.fedspeed_engine.is_in_stage_one:
            self.monitor.pause()
            self.evaluation()
            self.monitor.resume()

        # 回收显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 在每一步结束时，检查全局共识
        if self.early_stopper:
            self.early_stopper.check_global_consensus_and_stop(self.global_early_stop_votes)
                
    def _hook_on_fit_end(self, ctx):
        # --- 首先调用原有的 _hook_on_fit_end ---
        super()._hook_on_fit_end(ctx)
        # 在所有训练结束后，安全地停止 Reconstructor 的后台线程
        if self.fedspeed_engine and self.fedspeed_engine.reconstructor:
            self.fedspeed_engine.reconstructor.stop()
        if self.comm_executor:
            self.comm_executor.shutdown(wait=True)
            self.comm_executor = None

    def _trigger_stage_transition(self):
        """
        执行从阶段一到阶段二的转换。
        """
        logger.info(f"--- [Client #{self.client_id}] Triggering transition from Stage 1 to Stage 2 at step {self.fedspeed_engine.current_step} ---")
        
        # 1. 调用 Engine 执行全局模型同步
        self.fedspeed_engine.sync_model_at_stage_transition()
        
        # 2. (可选) 重置优化器状态
        #    因为模型参数发生了剧变，重置优化器状态（如动量）可能有助于稳定阶段二的训练
        # logger.info("Re-initializing optimizer for Stage 2.")
        # self.fedspeed_engine._initialize_optimizer()

        logger.info("--- Stage transition complete. Now entering Stage 2: Global Fine-tuning. ---")

    def passive_receive_message(self, message: Message):
        """
        一个被动的、轻量级的消息处理器。
        它的唯一职责就是将收到的消息放入内部队列。
        """
        self.internal_message_queue.put(message)

    def process_incoming_messages(self):
        """
        非阻塞地处理消息队列中所有待处理的消息。
        """
        # 初始化过程中接收到的消息
        if not self.internal_message_queue.empty():
            while not self.internal_message_queue.empty():
                try:
                    message = self.internal_message_queue.get_nowait()
                except queue.Empty:
                    break

                msg_type = message.msg_type
                
                # 加密的梯度分片消息
                if msg_type == 'anonymous_forward':
                    logger.debug("handle_anonymous_forward_message1")
                    self.handle_anonymous_message(message)
                # 非加密的参数分片更新消息
                elif msg_type == 'parameter_shard_update':
                    logger.debug("handle_parameter_shard_update_message1")
                    shard_list = message.content

                    # 将bytes转为torch
                    for shard_content in shard_list:
                        shard_data_list = shard_content.get('shard_data')
                        if isinstance(shard_data_list, dict) and "b64_pickled_tensor" in shard_data_list:
                            base64_string = shard_data_list["b64_pickled_tensor"]
                            pickled_bytes = base64.b64decode(base64_string.encode('ascii'))
                            shard_content['shard_data'] = pickle.loads(pickled_bytes)
                            
                            # 调用 Engine 更新
                    self.fedspeed_engine.update_parameter_shards(shard_list)
                # 早停投票消息
                elif msg_type == 'early_stop_vote':
                    sender_id = message.sender
                    vote = bool(message.content)
                    
                    if sender_id in self.global_early_stop_votes:
                        logger.info(f"Received early stop vote from Client #{sender_id}.")
                        self.global_early_stop_votes[sender_id] = vote
                # 非加密的模型消息
                elif msg_type == 'async_model_para' or msg_type == 'aggregated_model_para':
                    self.handle_model(message)
                else:
                    logger.debug(f"Trainer received an unhandled message type in async loop: '{msg_type}'")
        while True:
            # a. 从 CommManager 中非阻塞地拉取下一条消息
            message = self.comm_manager.receive_nowait()
            if message is None:
                # 消息队列空了，结束处理
                break
            msg_type = message.msg_type
            
            # 加密的梯度分片消息
            if msg_type == 'anonymous_forward':
                logger.debug("handle_anonymous_forward_message")
                self.handle_anonymous_message(message)
            # 非加密的参数分片更新消息
            elif msg_type == 'parameter_shard_update':
                logger.debug("handle_parameter_shard_update_message")
                shard_list = message.content

                # 将bytes转为torch
                for shard_content in shard_list:
                    shard_data_list = shard_content.get('shard_data')
                    if isinstance(shard_data_list, dict) and "b64_pickled_tensor" in shard_data_list:
                            base64_string = shard_data_list["b64_pickled_tensor"]
                            pickled_bytes = base64.b64decode(base64_string.encode('ascii'))
                            shard_content['shard_data'] = pickle.loads(pickled_bytes)
                self.fedspeed_engine.update_parameter_shards(shard_list)
            # 早停投票消息
            elif msg_type == 'early_stop_vote':
                sender_id = message.sender
                vote = bool(message.content)
                
                if sender_id in self.global_early_stop_votes:
                    logger.info(f"Received early stop vote from Client #{sender_id}.")
                    self.global_early_stop_votes[sender_id] = vote
            # 非加密的模型消息
            elif msg_type == 'async_model_para' or msg_type == 'aggregated_model_para':
                self.handle_model(message)
            else:
                logger.debug(f"Trainer received an unhandled message type in async loop: '{msg_type}'")

    def send_gradient_shards_anonymously_packaged(self, shard_list: list, target_client_id: int):
        """
        将一批发往同一个目标的梯度分片打包，并使用匿名路由一次性发送。
        """
        # --- 1. 手动将 Tensor 序列化为 bytes ---
        # 我们对原始的 shard_list 进行就地修改
        for content_dict in shard_list:
            tensor_data = content_dict.get('grad_shard_data')
            if isinstance(tensor_data, torch.Tensor):
                pickled_bytes = pickle.dumps(tensor_data)
                # 进行 Base64 编码，并解码为 ASCII 字符串
                base64_string = base64.b64encode(pickled_bytes).decode('ascii')
                content_dict['grad_shard_data'] = {"b64_pickled_tensor": base64_string}

        original_content = shard_list

        if self.anonymous_routing:
            # 调用 Router 构建并加密路由
            first_hop_id, final_message_content = self.router.build_anonymous_route(
                original_content, 
                target_client_id, 
                self.fedspeed_engine.current_step
            )

            if first_hop_id is not None and self.comm_executor:
                self.comm_executor.submit(
                    self._background_send_task,
                    'anonymous_forward',
                    [first_hop_id],
                    final_message_content
                )
            elif first_hop_id is not None:
                # 如果没有线程池，退化到同步发送
                self._background_send_task('anonymous_forward', [first_hop_id], final_message_content)
        else:
            self._background_send_task('anonymous_forward', [target_client_id], original_content)
            
    def broadcast_shard_updates(self, updated_shards: list):
        """
        将更新后的参数分片异步推送给其他客户端。
        """
        if not updated_shards:
            return

        # --- 1. 手动将 Tensor 序列化为 bytes ---
        for content_dict in updated_shards:
            tensor_data = content_dict.get('shard_data')
            if isinstance(tensor_data, torch.Tensor):
                pickled_bytes = pickle.dumps(tensor_data)
                base64_string = base64.b64encode(pickled_bytes).decode('ascii')
                content_dict['shard_data'] = {"b64_pickled_tensor": base64_string}

        # 决定接收者 (Gossip 或广播)
        if self.gossip_num >= 1 and len(self.peers) > 0:
            receivers = random.sample(self.peers, min(self.gossip_num, len(self.peers)))
        else:
            receivers = self.peers # 广播给所有邻居

        if not receivers:
            return

        if self.comm_executor:
            self.comm_executor.submit(
                self._background_send_task,
                'parameter_shard_update',
                receivers,
                updated_shards
            )
        else:
            # 退化到同步发送
            self._background_send_task('parameter_shard_update', receivers, updated_shards)

    def broadcast_early_stop_vote(self):
        """
        广播一个轻量级的、只包含早停投票的消息。
        """
        logger.info(f"Client #{self.client_id} is broadcasting its early stop vote.")
        
        # 接收者是所有其他客户端
        receivers = self.peers
        if not receivers: return

        message_content = 1
        if self.comm_executor:
             self.comm_executor.submit(
                self._background_send_task(
                    'early_stop_vote',
                    receivers,
                    message_content
                )
             )
        else:
            self._background_send_task('early_stop_vote', receivers, message_content)

        # 并且更新自己的全局状态，表示我已经投过票了
        self.global_early_stop_votes[self.client_id] = True

    def _background_send_task(self, msg_type: str, receivers: List[int], content: Any):
        """
        一个通用的、在后台线程中执行的发送任务。
        它可以直接访问 self.comm_manager 和 self.client_id。
        """
        try:
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.client_id,
                        receiver=receivers,
                        state=self.fedspeed_engine.current_step,
                        content=content)
            )
        except Exception as e:
            logger.error(f"Error in background send task for msg_type {msg_type}: {e}", exc_info=True)

    def _send_data(self, content: Any):
        """
        一个统一的数据发送器。处理 异步/同步, Gossip/广播, 匿名/非匿名 的发送逻辑。
        """
        # 对 Tensor 进行序列化
        if isinstance(content, torch.Tensor):
            pickled_bytes = pickle.dumps(content.cpu()) # 发送前移到 CPU
            message_content = base64.b64encode(pickled_bytes).decode('ascii')
        else:
            message_content = content.cpu()

        # 异步模式
        if self.asynchronous:
            # --- 确定接收者 (Gossip 或广播) ---
            receivers = []
            if self.gossip_num >= 1 and len(self.peers) > 0:
                # Gossip 模式
                receivers = random.sample(self.peers, min(self.gossip_num, len(self.peers)))
                logger.debug(f"Dispatching data to {len(receivers)} peers via Gossip: {receivers}")
            else: 
                # 广播模式
                receivers = self.peers
                logger.debug(f"Dispatching data to all {len(receivers)} peers via Broadcast.")
            
            if not receivers:
                logger.warning("No receivers found for dispatching data.")
                return

            # --- 判断是否使用匿名路由 ---
            if self.anonymous_routing:
                # 匿名模式
                if not self.router:
                    logger.error("Anonymous routing is enabled, but the router is not initialized.")
                    return

                # 对【每个】接收者，都构建一条独立的匿名路径
                for target_id in receivers:
                    # 调用 Router 构建并加密路由
                    first_hop_id, final_message_content = self.router.build_anonymous_route(
                        message_content,  # 加密提供的内容
                        target_id,
                        self.fedspeed_engine.current_step
                    )
                    
                    # 将加密后的消息发送给【路径的第一跳】
                    if first_hop_id:
                        self.comm_manager.send(
                            Message(msg_type='anonymous_forward',
                                    sender=self.client_id,
                                    receiver=[first_hop_id],
                                    state=self.fedspeed_engine.current_step,
                                    content=final_message_content)
                        )
            else:
                # 非匿名模式
                # 直接发送原始 content
                self.comm_manager.send(
                    Message(msg_type='async_model_para',
                            sender=self.client_id,
                            receiver=receivers,
                            state=self.fedspeed_engine.current_step,
                            content=message_content)
                )
        # 同步模式
        else:
            if self.aggregation_mode != "fl-sim" :
                self.comm_manager.send(
                    Message(msg_type='sync_model_para',
                            sender=self.client_id,
                            receiver=receivers,
                            state=self.current_step,
                            content=message_content)
                )
            else:
                self.comm_manager.send(
                    Message(msg_type='sync_model_para',
                            sender=self.client_id,
                            receiver=[0], # Server ID
                            state=self.current_step,
                            content=message_content)
                )

    def handle_anonymous_message(self, message: Message):
        """
        处理【加密】的匿名消息。
        """
        if self.anonymous_routing:
            # 调用 Router 处理
            result = self.router.handle_anonymous_message(message)
            if result is None: return
            action, data = result
        else:
            action = 'aggregate'

        if action == 'aggregate':
            logger.debug("recieve aggregate message")
            # 梯度分片
            if self.is_decentralized_sharded_mode:
                if self.anonymous_routing:
                    decrypted_content = data['content']
                else:
                    decrypted_content = message.content
                # 遍历并处理
                for item in decrypted_content:
                    # 将 bytes 转为 tensor
                    grad_data_list = item.get('grad_shard_data')
                    if isinstance(grad_data_list, dict) and "b64_pickled_tensor" in grad_data_list:
                        base64_string = grad_data_list["b64_pickled_tensor"]
                        pickled_bytes = base64.b64decode(base64_string.encode('ascii'))
                        item['grad_shard_data'] = pickle.loads(pickled_bytes)
                    self.fedspeed_engine.add_received_grad_shard(item)
            else:
                # a. Router 说我是最终目标，直接调用统一的核心处理器
                # 将 bytes 转为 tensor
                pickled_bytes = base64.b64decode(data['content'].encode('ascii'))
                decrypted_params_vec = pickle.loads(pickled_bytes)
                self._handle_incoming_model_vec(decrypted_params_vec.to(self.ctx.device), data['state'])
        elif action == 'forward':
            logger.debug("recieve forward message")
            # b. Router 说我是中继，需要转发
            self.comm_manager.send(
                Message(msg_type='anonymous_forward',
                        sender=self.client_id,
                        receiver=data['receiver'],
                        state=data['state'],
                        content=data['content'])
            )
        else:
            logger.warning(f"Trainer received an unhandled message type: '{action}'")

    def handle_model(self, message: Message):
        """
        处理【非加密】模型消息的公共入口。
        它的职责是【类型转换】(bytes -> tensor)，然后调用核心处理器。
        """
        # 1. 重建张量
        try:
            trainable_params = self.fedspeed_engine.local_optimizer.param_groups[0]['params']
            template_vec = parameters_to_vector([p.data for p in trainable_params])
            # 将 bytes 转为 tensor
            pickled_bytes = base64.b64decode(message.content.encode('ascii'))
            received_params_vec = pickle.loads(pickled_bytes)

            # 验证形状是否一致
            if received_params_vec.shape != template_vec.shape:
                logger.warning(f"Shape mismatch in received async model. "
                            f"Expected {template_vec.shape}, but got {received_params_vec.shape}. "
                            "Aggregation might be incorrect.")
        except Exception as e:
            logger.error(f"Failed to convert received bytes to tensor: {e}", exc_info=True)
            return
            
        # 2. 调用统一的核心处理器
        self._handle_incoming_model_vec(received_params_vec, message.state)

    def _handle_incoming_model_vec(self, received_params_vec: torch.Tensor, received_step: int):
        """
        一个统一的、私有的核心处理器，接收一个【torch.Tensor】和一个 step，然后执行聚合。
        """
        if not (hasattr(self, 'fedspeed_engine') and self.fedspeed_engine):
            return
        logger.debug(f"--- [FedSpeed-Model] Aggregating at Step #{self.current_step} ---")
            
        # 异步模式下，计算自适应权重
        if self.asynchronous and self.aggregation_mode != "fs-sim":
            alpha_for_this_aggregation = 0.5
            if self.adaptive_weight_cfg.get('use', False):
                alpha_for_this_aggregation = self._calculate_adaptive_alpha(received_step)
            
            # 调用 Engine 的异步聚合方法
            self.fedspeed_engine.aggregate_model(
                received_params_vec, 
                alpha_for_this_aggregation
            )
        else: # 同步模式
            # 调用 Engine 的同步更新方法
            self.fedspeed_engine.update_model(received_params_vec)

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

    def evaluation(self):
        """
        用于在训练过程中执行一次完整的评估。
        """
        logger.info(f"--- [Rank {self.fedspeed_engine.rank}] "
                    f"Starting evaluation at Step #{self.current_step} ---")
        
        # 1. 保存当前的模型状态 (train 模式)
        is_training_mode = self.ctx.model.training
        self.ctx.model.eval() # 切换到评估模式

        # 2. 遍历所有需要评估的数据集 (例如 ['val', 'test'])
        # for split in self.cfg.eval.split:    
        split = 'val'        
        # a. 准备数据加载器，确保 loader 存在且可迭代
        ctx = self.ctx
        loader_key = f"{split}_loader"
        if ctx.get(loader_key) is not None:
            # 假设test和val数据集大小一致
            if self.num_batches == 0:
                self.num_batches = len(ctx.get(loader_key))
            if not isinstance(ctx.get(loader_key), ReIterator):
                setattr(ctx, loader_key, ReIterator(ctx.get(loader_key)))
            
            # 获取最终的 dataloader
            dataloader = ctx.get(loader_key)
            dataloader.reset() # 确保从头开始迭代
            
            # b. 前向传播计算loss
            total_loss = 0.0
            total_samples = 0
            nan_batches = 0 # <-- 新增：记录 NaN batch 的数量

            with torch.no_grad():
                for i in range(self.num_batches):
                    try:
                        batch_data = next(dataloader)
                        ctx.data_batch = batch_data
                        
                        data_batch_on_device = {k: v.to(ctx.device) for k, v in ctx.data_batch.items()}
                        # 复制与 _hook_on_batch_forward 中相同的修复逻辑
                        # 检查 attention_mask 的类型，并进行转换以兼容 adapters 库
                        if 'attention_mask' in data_batch_on_device and \
                        data_batch_on_device['attention_mask'].dtype == torch.bool:
                            
                            logger.debug("[Evaluation] Converting boolean attention_mask to long for adapter compatibility.")
                            data_batch_on_device['attention_mask'] = data_batch_on_device['attention_mask'].long()
                        outputs = self.fedspeed_engine.forward(**data_batch_on_device)
                        
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            # --- 关键修复 2: 在评估循环内部检查并跳过 NaN loss ---
                            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                                nan_batches += 1
                                logger.warning(
                                    f"[Evaluation] Loss is NaN/Inf on batch {i+1}/{self.num_batches}. Skipping this batch."
                                )
                                continue # 跳过这个batch，不计入 total_loss
                            # --- 结束修复 2 ---
                            
                            batch_loss = outputs.loss.item()
                            batch_size = data_batch_on_device['input_ids'].size(0)
                            total_loss += batch_loss * batch_size
                            total_samples += batch_size
                    except StopIteration:
                        break
            
            # 报告 NaN batch 的情况
            if nan_batches > 0:
                logger.warning(
                    f"[Evaluation] Encountered and skipped {nan_batches} batches with NaN/Inf loss "
                    f"out of {self.num_batches} total batches in the validation set."
                )

            # 即使所有batch都是NaN，也要避免除以零
            if total_samples == 0:
                # 如果所有batch都被跳过了，avg_loss 应该是 NaN 或 inf，
                # 这样 EarlyStopper 就能正确地将其识别为一次失败的评估。
                avg_loss = float('nan') 
            else:
                avg_loss = total_loss / total_samples

            eval_results = {
                f'{split}_avg_loss': avg_loss,
                f'{split}_total': total_samples,
                f'{split}_nan_batches': nan_batches # <-- 新增：在结果中也记录下来
            }
            logger.info(f"  - Evaluation results on '{split}': {eval_results}")

            if self.early_stopper:
                # --- 关键修复 3: 将 NaN loss 传递给 EarlyStopper ---
                # EarlyStopper 需要能够处理 NaN 值，将其视为最差表现
                score_to_check = avg_loss
                if score_to_check is None or np.isnan(score_to_check) or np.isinf(score_to_check):
                    # 如果 avg_loss 是无效值, 我们给一个很大的惩罚值，确保它不会被认为是最佳分数
                    # 或者让 EarlyStopper 内部处理它
                    pass # 让 EarlyStopper 接收 NaN
                
                self.early_stopper(score_to_check, self.fedspeed_engine)
                if self.early_stopper.early_stop and not self.global_early_stop_votes[self.client_id]:
                    self.monitor.self_stop()
                    # 广播早停投票
                    logger.info(f"--- [Client {self.client_id}] local early stop condition TRIGGERED! Broadcast vote. ---")
                    self.broadcast_early_stop_vote()
        else:
            logger.warning(f"Dataloader for split '{split}' not found in trainer or ctx. Skipping evaluation.")

        # 3. 恢复模型到之前的状态
        if is_training_mode:
            self.ctx.model.train()

    def initialize_router(self, client_id):
        """
        一个由 Client 调用的、专门用于初始化 Router 的方法。
        """
        if self.anonymous_routing and self.router is None:
             logger.info(f"Client #{client_id}: Initializing AnonymousRouter.")
             self.router = AnonymousRouter(
                 client_id=client_id,
                 cfg=self.cfg
             )

def apply_checkpointing(model, block_class, engine):
    """
    遍历模型，用一个包装了 checkpoint 的新 forward 方法来替换掉原始的 forward 方法。
    """
    logger.info(f"Applying activation checkpointing to all modules of type {block_class.__name__}")
    
    engine.use_checkpointing = True

    for module_to_patch in model.modules():
        if isinstance(module_to_patch, block_class):
            module_to_patch.original_forward = module_to_patch.forward
            
            def new_forward(*args, m=module_to_patch, **kwargs):
                # Checkpointing 只在训练模式下生效
                if not m.training:
                    return m.original_forward(*args, **kwargs)

                # 定义在 recomputation 期间要执行的函数
                def recompute_function(*args, **kwargs):
                    # 在重新计算前，设置标志位
                    engine.is_recomputing = True
                    try:
                        result = m.original_forward(*args, **kwargs)
                    finally:
                        # 确保在计算结束后，总是恢复标志位
                        engine.is_recomputing = False
                    return result

                # 使用 checkpoint 调用这个包装了状态管理的函数
                return checkpoint(
                    recompute_function,
                    *args,
                    use_reentrant=False,
                    **kwargs
                )
            
            module_to_patch.forward = new_forward

def call_fedspeed_trainer(trainer_type):
    if trainer_type == 'fedspeedtrainer':
        trainer_builder = FedSpeedTrainer
        return _trainer_builder
        
register_trainer('fedspeedtrainer', call_fedspeed_trainer)
