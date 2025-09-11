import logging
import torch
import torch.distributed as dist
import random
import os

from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.trainers.context import CtxVar
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.utils.checkpoint import checkpoint
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.llm.trainer.fedspeed_engine import FedSpeedEngine, log_memory_usage
from federatedscope.register import register_trainer
from federatedscope.core.message import Message
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from federatedscope.core.auxiliaries.ReIterator import ReIterator

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
        在每次评估后调用此方法。
        
        Args:
            current_score (float): 当前的评估分数 (我们用 loss，所以越小越好)。
            engine (FedSpeedEngine): FedSpeed 引擎实例，用于保存模型。
        """
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(engine)
        
        elif current_score > self.best_score + self.delta:
            # 性能变差了
            self.counter += 1
            logger.info(f"[EarlyStopper] No improvement. Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                raise EarlyStopException()
        else:
            # 性能提升了
            self.best_score = current_score
            self.save_checkpoint(engine)
            self.counter = 0

    def save_checkpoint(self, engine: FedSpeedEngine):
        """保存当前最佳的模型参数。"""
        # 目前单机只让 Rank 0 保存，后续多机修改
        if engine.rank == 0: 
            logger.info(f"[EarlyStopper] New best score: {self.best_score:.4f}. Saving model state...")
            best_params_vec = engine.get_full_trainable_params_as_vector()
            torch.save(best_params_vec, self.checkpoint_path)

    def load_best_checkpoint(self, engine: FedSpeedEngine):
        """将模型恢复到最佳状态。"""
        if os.path.exists(self.checkpoint_path):
            best_params_vec = torch.load(self.checkpoint_path, map_location=engine.device)
            engine.set_full_trainable_params_from_vector(best_params_vec)
            logger.info(f"Rank {engine.rank}: Successfully loaded best model state.")


class EarlyStopException(Exception):
    pass


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
        self.aggregation_mode = self.config.federate.get('aggregation_mode') 
        self.aggregation_steps = self.config.federate.get('aggregation_steps') 
        self.asynchronous = self._cfg.federate.get('asynchronous_aggregation', False)
        self.adaptive_weight_cfg = self.cfg.federate.get('adaptive_weight', {'use': False})
        self.gossip_num = self._cfg.federate.get('gossip_num', 1)
        self.eval_freq = self._cfg.eval.get('freq')
        if self.cfg.early_stop.patience > 0:
            self.early_stopper = EarlyStopper(
                patience=self.cfg.early_stop.patience,
                checkpoint_path=os.path.join(self.cfg.outdir, "fedspeed_best_model.ckpt")
            )
        else:
            self.early_stopper = None


    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        # 检查并创建FedSpeedEngine
        if self.fedspeed_engine is None and dist.is_initialized():
            if self.fedspeed_init_package is None:
                raise RuntimeError("FedSpeed Trainer started but init package was not set by the client.")

            init_package = self.fedspeed_init_package
            self.aggregation_mode = self.config.federate.get('aggregation_mode') 
            if self.aggregation_mode == 'local-reconstruct':
                self.fedspeed_engine = FedSpeedEngine(
                ctx.model, 
                self.cfg, 
                ctx.device, 
                dist.group.WORLD,
                initial_shards=init_package,
                ref_counts=init_package['ref_counts']
            )
            else:
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

            # 在所有训练开始前，启动 Reconstructor 的后台线程
            if self.fedspeed_engine and self.fedspeed_engine.reconstructor:
                self.fedspeed_engine.reconstructor.start()

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
            return initial_alpha * (decay_rate ** staleness)
        elif func_type == 'linear':
            decay_factor = self.adaptive_weight_cfg.get('decay_factor', 0.01)
            return max(0.0, initial_alpha - staleness * decay_factor)
        # ... (可以扩展其他函数)
        else: # 'constant'
            return initial_alpha

    # 将所有复杂逻辑都委托给 Engine
    def _hook_on_batch_forward(self, ctx):
        # 调用引擎的forward方法
        #    所有的All-Gather和释放都会在内部被钩子自动处理
        data_batch_on_device = {k: v.to(ctx.device) for k, v in ctx.data_batch.items()}
        log_memory_usage("Forward Start")
        outputs = self.fedspeed_engine.forward(**data_batch_on_device)
        log_memory_usage("Forward End")
        
        # 将结果保存到 ctx 中
        #    假设模型在训练时返回了loss（当提供了labels时）
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
        self.current_step += 1

        # 1. 聚合梯度
        log_memory_usage("Aggregate Grads Start")            
        aggregated_grads_dict = self.fedspeed_engine.aggregate_gradients()
        log_memory_usage("Aggregate Grads End (Full Grads Cleared)")
        # 2. 执行优化器步骤
        log_memory_usage("Optimizer Step Start")
        params_to_send = self.fedspeed_engine.step(aggregated_grads_dict)
        log_memory_usage("Optimizer Step End")

        # 3. 模型聚合
        if params_to_send is not None:
            if self.asynchronous:
                receivers = []
                if self.gossip_num >= 1:
                    # --- Gossip 模式 ---
                    # 从邻居列表中，无放回地随机抽取 gossip_num 个
                    if len(self.peers) > 0:
                        receivers = random.sample(self.peers, min(self.gossip_num, len(self.peers)))
                    logger.debug(f"Gossip send at Step #{self.current_step} to peers: {receivers}")
                else:
                    # --- 原有的广播模式 ---
                    receivers = self.peers
                    logger.debug(f"Broadcast send at Step #{self.current_step} to all peers.")
                self.comm_manager.send(
                    Message(msg_type='async_model_para',
                            sender=self.client_id,
                            receiver=receivers,
                            state=self.current_step,
                            content=params_to_send.cpu()) 
                )
            else:
                self.comm_manager.send(
                     Message(msg_type='sync_model_para',
                            sender=self.client_id,
                            receiver=[0], # Server ID
                            state=self.current_step,
                            content=params_to_send.cpu())
                )
        if self.aggregation_mode != 'gradient':
            if self.asynchronous:
                while True:
                    message_list = self.comm_manager.receive_nowait()
                    if message_list is None:
                        # 缓冲区空了，立即退出循环
                        break
                    for message in message_list:
                        self.handle_model(message)
                    break
            elif self.fedspeed_engine.should_aggregate:
                message = self.comm_manager.receive()
                self.handle_model(message)

        # 4. 打印日志
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        logger.debug(f"--- Rank {my_rank} | Step {self.current_step} | Loss: {ctx.loss_batch.item():.4f} ---")

        # 5. 更新FederatedScope的统计变量
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

        # 6. 执行评估
        if self.eval_freq > 0 and (self.current_step) % self.eval_freq == 0:
            self.evaluation()


    def handle_model(self, message):
        """
        这个方法作为消息处理器，由 Client 的事件循环调用。
        """
        logger.info(f"--- [FedSpeed-Model] Aggregating at Step #{self.current_step} ---")

        if self.fedspeed_engine is None: return
        
        # 1. 获取一个【模板张量】，以便知道正确的 shape, dtype
        trainable_params = self.fedspeed_engine.local_optimizer.param_groups[0]['params']
        template_vec = parameters_to_vector([p.data for p in trainable_params])

        # 2. 获取原始的数据列表
        received_data_list = message.content
        received_step = message.state

        # 3. 使用模板张量的元数据，从列表创建新的张量
        try:
            received_params_vec = torch.tensor(
                received_data_list, 
                dtype=template_vec.dtype # 确保数据类型一致
            ).to(self.ctx.device)
            
            # d. 验证形状是否一致
            if received_params_vec.shape != template_vec.shape:
                logger.warning(f"Shape mismatch in received async model. "
                            f"Expected {template_vec.shape}, but got {received_params_vec.shape}. "
                            "Aggregation might be incorrect.")

        except Exception as e:
            logger.error(f"Failed to convert received list to tensor: {e}", exc_info=True)
            return
            
        if self.asynchronous:
            if self.adaptive_weight_cfg['use']:
                alpha_for_this_aggregation = self._calculate_adaptive_alpha(received_step)
                logger.debug(f"alpha_for_this_aggregation: {alpha_for_this_aggregation}")
            if hasattr(self, 'fedspeed_engine') and self.fedspeed_engine:
                # 调用 Engine 的聚合方法，这个方法内部需要加锁
                self.fedspeed_engine.aggregate_model(received_params_vec, alpha_for_this_aggregation)
        else:
            self.fedspeed_engine.update_model(received_params_vec)

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
            with torch.no_grad():
                for _ in range(self.num_batches):
                    try:
                        # i. 准备 batch 数据
                        batch_data = next(dataloader)
                        ctx = self.ctx # 复用 trainer 的上下文
                        ctx.data_batch = batch_data
                        
                        # ii. 执行前向传播
                        data_batch_on_device = {k: v.to(ctx.device) for k, v in ctx.data_batch.items()}
                        # Engine 的 forward 逻辑 (包括 hooks) 会自动处理参数重建
                        outputs = self.fedspeed_engine.forward(**data_batch_on_device)
                        
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            batch_loss = outputs.loss.item()
                            batch_size = data_batch_on_device['input_ids'].size(0)
                            total_loss += batch_loss * batch_size
                            total_samples += batch_size
                    except StopIteration:
                        # 正常情况下不应发生，但作为保护
                        break
                
                # c. 计算全局平均损失
                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

                # d. 打印评估结果
                eval_results = {
                    f'{split}_avg_loss': avg_loss,
                    f'{split}_total': total_samples,
                }
                logger.info(f"  - Evaluation results on '{split}': {eval_results}")

                # e. 判断是否早停
                if self.early_stopper:
                    self.early_stopper(avg_loss, self.fedspeed_engine)
        else:
            logger.warning(f"Dataloader for split '{split}' not found in trainer or ctx. Skipping evaluation.")

        # 3. 恢复模型到之前的状态
        if is_training_mode:
            self.ctx.model.train()

    def _hook_on_fit_end(self, ctx):
        # --- 首先调用原有的 _hook_on_fit_end ---
        super()._hook_on_fit_end(ctx)
        # 在所有训练结束后，安全地停止 Reconstructor 的后台线程
        if self.fedspeed_engine and self.fedspeed_engine.reconstructor:
            self.fedspeed_engine.reconstructor.stop()

def apply_checkpointing(model, block_class, engine):
    """
    遍历模型，用一个包装了 checkpoint 的新 forward 方法
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
