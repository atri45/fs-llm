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
from federatedscope.llm.trainer.anonymous_router import AnonymousRouter

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
        self.router = None

        self.aggregation_mode = self.config.federate.get('aggregation_mode') 
        self.aggregation_steps = self.config.federate.get('aggregation_steps') 
        self.asynchronous = self._cfg.federate.get('asynchronous_aggregation', False)
        self.adaptive_weight_cfg = self.cfg.federate.get('adaptive_weight', {'use': False})
        self.gossip_num = self._cfg.federate.get('gossip_num', 1)
        self.eval_freq = self._cfg.eval.get('freq')
        self.anonymous_routing = self.cfg.federate.get('anonymous_routing', False)
        
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
            if self.asynchronous and self.anonymous_routing:
                # --- A. 异步 + 匿名路由模式 ---
                if not self.router:
                    logger.error("Anonymous routing is enabled, but the router is not initialized.")
                    return
                
                # 1. 随机选择一个【最终目标】客户端
                if not self.peers:
                    logger.warning("No peers to send anonymous message to. Skipping.")
                    return
                target_id = random.choice(self.peers)
                
                # 2. 调用 Router 构建并加密路由
                first_hop_id, final_message_content = self.router.build_anonymous_route(
                    params_to_send, target_id, self.current_step
                )
                
                # 3. 将加密后的消息发送给【路径的第一跳】
                if first_hop_id:
                    self.comm_manager.send(
                        Message(msg_type='anonymous_forward',
                                sender=self.client_id,
                                receiver=[first_hop_id],
                                state=self.current_step,
                                content=final_message_content)
                    )
            elif self.asynchronous:
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

        # 4. 处理消息队列
        if self.aggregation_mode != 'gradient':
            if self.asynchronous:
                while True:
                    # a. 从 CommManager 中非阻塞地拉取下一条消息
                    message = self.comm_manager.receive_nowait()
                    if message is None:
                        # 消息队列空了，结束处理
                        break

                    # b. 根据消息类型进行分发
                    msg_type = message.msg_type
                    if msg_type == 'async_model_para' or msg_type == 'aggregated_model_para':
                        # --- 是一个【非加密】的模型消息 ---
                        self.handle_model(message)
                    elif msg_type == 'anonymous_forward':
                        # --- 是一个【加密】的匿名消息 ---
                        self.handle_anonymous_message(message)
            elif self.fedspeed_engine.should_aggregate:
                message = self.comm_manager.receive()
                self.handle_model(message)

        # 5. 打印日志
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        logger.debug(f"--- Rank {my_rank} | Step {self.current_step} | Loss: {ctx.loss_batch.item():.4f} ---")

        # 6. 更新FederatedScope的统计变量
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

        # 7. 执行评估
        if self.eval_freq > 0 and (self.current_step) % self.eval_freq == 0:
            self.evaluation()

    def _hook_on_fit_end(self, ctx):
        # --- 首先调用原有的 _hook_on_fit_end ---
        super()._hook_on_fit_end(ctx)
        # 在所有训练结束后，安全地停止 Reconstructor 的后台线程
        if self.fedspeed_engine and self.fedspeed_engine.reconstructor:
            self.fedspeed_engine.reconstructor.stop()

    def handle_anonymous_message(self, message: Message):
        """
        处理【加密】的匿名消息。
        """
        if not self.router: return
        
        # 调用 Router 处理
        result = self.router.handle_anonymous_message(message)
        if result is None: return

        action, data = result
        if action == 'aggregate':
            # a. Router 说我是最终目标，直接调用统一的核心处理器
            decrypted_params_vec = data['content']
            decrypted_step = data['state']
            self._handle_incoming_model_vec(decrypted_params_vec.to(self.ctx.device), decrypted_step)
        elif action == 'forward':
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
        它的职责是【类型转换】(list -> tensor)，然后调用核心处理器。
        """
        received_data_list = message.content
        received_step = message.state

        # 1. 重建张量
        try:
            trainable_params = self.fedspeed_engine.local_optimizer.param_groups[0]['params']
            template_vec = parameters_to_vector([p.data for p in trainable_params])
            received_params_vec = torch.tensor(
                received_data_list, 
                dtype=template_vec.dtype
            ).to(self.ctx.device)

            # 验证形状是否一致
            if received_params_vec.shape != template_vec.shape:
                logger.warning(f"Shape mismatch in received async model. "
                            f"Expected {template_vec.shape}, but got {received_params_vec.shape}. "
                            "Aggregation might be incorrect.")
        except Exception as e:
            logger.error(f"Failed to convert received list to tensor: {e}", exc_info=True)
            return
            
        # 2. 调用统一的核心处理器
        self._handle_incoming_model_vec(received_params_vec, received_step)

    def _handle_incoming_model_vec(self, received_params_vec: torch.Tensor, received_step: int):
        """
        一个统一的、私有的核心处理器，接收一个【torch.Tensor】和一个 step，然后执行聚合。
        """
        if not (hasattr(self, 'fedspeed_engine') and self.fedspeed_engine):
            return
        logger.info(f"--- [FedSpeed-Model] Aggregating at Step #{self.current_step} ---")
            
        # 异步模式下，计算自适应权重
        if self.asynchronous:
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

    def initialize_router(self, client_id, peers):
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
