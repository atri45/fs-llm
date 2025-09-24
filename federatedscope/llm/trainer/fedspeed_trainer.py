import logging
import pickle
import base64
import torch
import torch.distributed as dist
import random
import os
import queue

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
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(engine)
        
        elif current_score > self.best_score + self.delta:
            # 性能变差了
            self.counter += 1
            logger.info(f"[EarlyStopper] No improvement. Counter: {self.counter} / {self.patience}")
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

        self.aggregation_mode = self.config.federate.get('aggregation_mode') 
        self.is_decentralized_sharded_mode = (self.aggregation_mode == 'shardedGradient')
        self.aggregation_steps = self.config.federate.get('aggregation_steps') 
        self.stage_one_steps = self.cfg.federate.two_stages.get('stage_one_steps', 250)
        self.asynchronous = self._cfg.federate.get('asynchronous_aggregation', False)
        self.adaptive_weight_cfg = self.cfg.federate.get('adaptive_weight', {'use': False})
        self.gossip_num = self._cfg.federate.get('gossip_num', 1)
        self.eval_freq = self._cfg.eval.get('freq')
        self.anonymous_routing = self.cfg.federate.get('anonymous_routing', False)
        if self.is_decentralized_sharded_mode:
            self.anonymous_routing = True
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

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        # 检查并创建FedSpeedEngine
        if self.fedspeed_engine is None and dist.is_initialized():
            if self.fedspeed_init_package is None:
                raise RuntimeError("FedSpeed Trainer started but init package was not set by the client.")

            init_package = self.fedspeed_init_package
            self.aggregation_mode = self.config.federate.get('aggregation_mode') 
            if self.aggregation_mode in ['shardedGradient', 'local-reconstruct']:
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

    # 将所有复杂逻辑都委托给 Engine
    def _hook_on_batch_forward(self, ctx):
        logger.debug(f"batch_forward")
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
        logger.debug(f"batch_backward")
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
        self.fedspeed_engine.current_step += 1

        # --- 检查是否需要阶段转换 ---
        if self.fedspeed_engine.noniid and self.current_step == self.stage_one_steps:
            self._trigger_stage_transition()

        # 通用异步逻辑 (消息处理)
        logger.debug(f"process_incoming_messages")
        # if self.asynchronous:
        self.process_incoming_messages()    

        # 梯度分片处理流程
        if self.is_decentralized_sharded_mode:
            self.fedspeed_engine.current_step += 1
            
            # 1. 处理本地计算的梯度：保留自己的分片，准备并匿名发送其他人的分片
            logger.debug(f"process_and_dispatch_gradients")
            my_local_grad_shards, grouped_shards_to_dispatch = self.fedspeed_engine.process_and_dispatch_gradients()

            # 2. 匿名发送梯度分片
            if grouped_shards_to_dispatch:
                logger.debug(f"send_gradient_shard_anonymously")
                for target_client_id, shard_list in grouped_shards_to_dispatch.items():
                    if shard_list: # 确保列表不为空
                        self.send_gradient_shards_anonymously_packaged(
                            shard_list, # 发送整个列表
                            target_client_id
                        )

            # 3. 执行优化器步骤 (内部包含梯度 FedAvg 聚合)
            logger.debug(f"sharded_step")
            updated_shards_for_broadcast, model_was_updated_by_peers = self.fedspeed_engine.sharded_step(my_local_grad_shards)

            # 4. 异步推送更新后的参数分片 (非匿名)
            if model_was_updated_by_peers and self.fedspeed_engine.current_step > self.fedspeed_engine.stage_one_steps:
                logger.debug(f"broadcast_shard_updates")
                self.broadcast_shard_updates(updated_shards_for_broadcast)

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
                self._send_data(params_to_send)

                # 4. 等待并处理同步消息
                if self.aggregation_mode == 'fl-sim':
                    message = self.comm_manager.receive()
                    if message.msg_type == 'aggregated_model_para':
                        self.handle_model(message)

        # 日志和评估
        my_rank = dist.get_rank()
        logger.info(f"--- Rank {my_rank} | Step {self.current_step} | Loss: {ctx.loss_batch.item():.4f} ---")
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        stage_one_steps = self.cfg.federate.get('stage_one_steps', 0)
        if self.eval_freq > 0 and (self.current_step % self.eval_freq) == 0 and self.current_step > stage_one_steps:
            self.evaluation()

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
        if not self.router:
            logger.error("Anonymous routing is enabled, but the router is not initialized.")
            return

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
        if not self.router: return
        
        # 调用 Router 处理
        result = self.router.handle_anonymous_message(message)
        if result is None: return

        action, data = result
        if action == 'aggregate':
            # 梯度分片
            if self.is_decentralized_sharded_mode:
                decrypted_content = data['content']
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
        logger.info(f"--- [FedSpeed-Model] Aggregating at Step #{self.current_step} ---")
            
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
                    if self.early_stopper.early_stop and not self.global_early_stop_votes[self.client_id]:
                        # 广播早停投票
                        logger.info(f"--- [Client {self.client_id}] local early stop condition TRIGGERED! Broadcast vote. ---")
                        self.broadcast_early_stop_vote()
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
