import logging
import torch
import torch.distributed as dist

from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.trainers.context import CtxVar
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.utils.checkpoint import checkpoint
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.llm.trainer.fedspeed_engine import FedSpeedEngine, log_memory_usage
from federatedscope.register import register_trainer

logger = logging.getLogger(__name__)

class FedSpeedTrainer(LLMTrainer):
    def __init__(self, model, data, device, config, **kwargs):
        super().__init__(model, data, device, config, **kwargs)
        self.fedspeed_engine = None
        self.fedspeed_init_package = None
        self.current_step = 0
        self.eval_freq = self.cfg.eval.freq

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        # 检查并创建FedSpeedEngine
        if self.fedspeed_engine is None and dist.is_initialized():
            if self.fedspeed_init_package is None:
                raise RuntimeError("FedSpeed Trainer started but init package was not set by the client.")

            init_package = self.fedspeed_init_package
            self.fedspeed_engine = FedSpeedEngine(
                ctx.model, 
                self.cfg, 
                ctx.device, 
                dist.group.WORLD,
                initial_shards=init_package['shards'],
                ref_counts=init_package['ref_counts']
            )
            
            # 激活检查点和钩子注册逻辑不变
            if self.cfg.federate.use_activation_checkpointing:
                apply_checkpointing(ctx.model, GPT2Block, self.fedspeed_engine)
                self.fedspeed_engine._register_hooks_recursively(self.fedspeed_engine.model)
            
            # 清理，防止内存泄漏
            self.fedspeed_init_package = None

    # 我们将所有复杂逻辑都委托给 Engine
    def _hook_on_batch_forward(self, ctx):
        # ordered_trainable_names = sorted(list(self.fedspeed_engine.trainable_param_names))
        # with torch.no_grad():
        #     for name in ordered_trainable_names:
        #         param = self.fedspeed_engine.name_to_param_map[name]
        #         # 手动 all-gather
        #         shard_list = [torch.empty_like(param.fedspeed_shard) for _ in range(self.fedspeed_engine.world_size)]
        #         dist.all_gather(shard_list, param.fedspeed_shard)
        #         full_param_data = torch.cat(shard_list, dim=0)
        #         norm = torch.linalg.norm(full_param_data.float()).item()
        #         logger.info(f"  - Norm of '{name}': {norm:.8f}")
        # # --- 打印逻辑结束 ---

        # 调用引擎的forward方法
        #    所有的All-Gather和释放都会在内部被钩子自动处理
        data_batch_on_device = {k: v.to(ctx.device) for k, v in ctx.data_batch.items()}
        log_memory_usage("Forward Start")
        outputs = self.fedspeed_engine.forward(**data_batch_on_device)
        log_memory_usage("Forward End")
        
        # 将结果保存到 ctx 中
        #    我们假设模型在训练时返回了loss（当提供了labels时）
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            ctx.loss_batch = CtxVar(outputs.loss, "batch")
            ctx.loss_task = CtxVar(outputs.loss, "batch") # 简化
            ctx.skip_this_batch = CtxVar(False, "batch")
        else:
            # 处理推理或未提供label的情况
            ctx.skip_this_batch = CtxVar(True, "batch")
            logger.warning("No loss returned from model forward pass. Skipping backward.")
        
        # ... 其他 ctx 变量的赋值，例如 logits
        if hasattr(outputs, 'logits'):
            ctx.y_prob = CtxVar(outputs.logits, "batch")
        if 'labels' in ctx.data_batch:
            ctx.y_true = CtxVar(ctx.data_batch['labels'], "batch")
        ctx.batch_size = CtxVar(len(ctx.data_batch['input_ids']), "batch")

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return
        if hasattr(ctx, 'loss_task') and ctx.loss_task.requires_grad:
            # 这一步只负责调用 engine.backward()，它会完成流式的参数管理和梯度收集
            log_memory_usage("Backward Start")
            self.fedspeed_engine.backward(ctx.loss_task)
            log_memory_usage("Backward End (Grads Collected)")

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            # 在跳过时，也要确保清理可能存在的状态
            if hasattr(self.fedspeed_engine, 'temp_full_gradients'):
                self.fedspeed_engine.temp_full_gradients.clear()
            return

        # 1. 聚合梯度
        log_memory_usage("Aggregate Grads Start")            
        aggregated_grads_dict = self.fedspeed_engine.aggregate_gradients()
        log_memory_usage("Aggregate Grads End (Full Grads Cleared)")
        # 2. 执行优化器步骤
        log_memory_usage("Optimizer Step Start")
        self.fedspeed_engine.step(aggregated_grads_dict)
        log_memory_usage("Optimizer Step End")
        # 3. 更新FederatedScope的统计变量
        self.current_step += 1
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

        # 4. 打印日志
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(f"--- Rank {my_rank} | Step {self.current_step} | Loss: {ctx.loss_batch.item():.4f} ---")

def apply_checkpointing(model, block_class, engine):
    """
    [修正版] 遍历模型，用一个包装了 checkpoint 的新 forward 方法
    来替换掉原始的 forward 方法。使用默认参数来正确捕获闭包变量。
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
                    # 【核心】在重新计算前，设置标志位
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
