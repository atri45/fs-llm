import torch
import logging
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.llm.model.adapter_builder import AdapterModel
from torch.nn.utils.convert_parameters import parameters_to_vector

logger = logging.getLogger(__name__)


class LLMTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):

        # logger.info(f"--- [FL Client #{ctx.cfg.distribute.data_idx}] "
        #             f"Params BEFORE FORWARD at Round ---")
        # with torch.no_grad():
        #     for name, param in ctx.model.named_parameters():
        #         if param.requires_grad:
        #             norm = torch.linalg.norm(param.data.float()).item()
        #             logger.info(f"  - Norm of '{name}': {norm:.8f}")
        # # --- 打印逻辑结束 ---

        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        attention_mask = attention_mask.long()
        log_memory_usage("Forward Start")
        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)
        log_memory_usage("Forward End")
        logits = outputs.logits
        loss = outputs.loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.deepspeed.use:
            log_memory_usage("Backward Start")
            ctx.model_engine.backward(ctx.loss_task)
            log_memory_usage("Backward End (Grads Collected)")
            log_memory_usage("Step Start")
            ctx.model_engine.step()
            log_memory_usage("Step End")
        else:
            ctx.optimizer.zero_grad()
            log_memory_usage("Backward Start")
            real_lr = 0.001
            ctx.loss_task.backward()

            # # --- 在这里打印梯度 ---
            # logger.info(f"--- [FL Trainer] Gradients at Round #---")
            # total_grad_norm = 0.0
            # # 我们只打印可训练的 (LoRA) 参数的梯度
            # for name, param in ctx.model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         grad_norm = torch.linalg.norm(param.grad.detach().float()).item()
            #         total_grad_norm += grad_norm ** 2
            #         # 打印每个 LoRA 参数梯度的 L2 范数，这是一个很好的摘要信息
            #         logger.info(f"  - Grad norm of '{name}': {grad_norm:.6f}")
            # total_grad_norm = total_grad_norm ** 0.5
            # logger.info(f"  - TOTAL GRAD NORM (L2): {total_grad_norm:.6f}")
            # # --- 打印逻辑结束 ---

            # with torch.no_grad():
            #     for param in ctx.model.parameters():
            #         if param.grad is not None:
            #             param.grad.mul_(real_lr)
            log_memory_usage("Backward End (Grads Collected)")
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
            log_memory_usage("Step Start")
            ctx.optimizer.step()

            # # --- 打印【本地更新后】的参数范数 ---
            # total_norm_sq = 0.0
            # with torch.no_grad():
            #     # 我们只打印可训练的 (LoRA) 参数
            #     for name, param in ctx.model.named_parameters():
            #         if param.requires_grad:
            #             norm = torch.linalg.norm(param.data.float()).item()
            #             total_norm_sq += norm ** 2
            #             logger.info(f"  - Norm of '{name}': {norm:.8f}")
            # logger.info(f"  - TOTAL PARAMS NORM: {total_norm_sq:.8f}")
            # --- 打印逻辑结束 ---
        #     # --- 在 step() 之后打印优化器状态 ---
        #     logger.info(f"--- [FL Trainer] Optimizer State after Step at Round ---")
        #     total_state_mem = 0
            
        #     # 检查 state 是否为空
        #     if not ctx.optimizer.state:
        #         logger.info("  - Optimizer state is empty.")
            
        #     # 遍历优化器状态字典
        #     for param, state in ctx.optimizer.state.items():
        #         # 找到这个参数的名称，以便更好地识别
        #         param_name = "Unknown"
        #         for name, p in ctx.model.named_parameters():
        #             if id(p) == id(param):
        #                 param_name = name
        #                 break

        #         logger.info(f"  - State for param '{param_name}':")
        #         for key, value in state.items():
        #             if isinstance(value, torch.Tensor):
        #                 mem_bytes = value.numel() * value.element_size()
        #                 total_state_mem += mem_bytes
        #                 norm = torch.linalg.norm(value.float()).item()
        #                 logger.info(f"    - '{key}': shape={list(value.shape)}, norm={norm:.6f}, mem={mem_bytes/1024:.2f} KB")
        #             else:
        #                 logger.info(f"    - '{key}': {value}")
            
        #     logger.info(f"  - TOTAL OPTIMIZER STATE MEMORY: {total_state_mem / (1024**2):.4f} MB")
        # # --- 打印逻辑结束 ---
            log_memory_usage("Step End")
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        setattr(ctx, 'eval_metrics', eval_results)

        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size
            
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

def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
