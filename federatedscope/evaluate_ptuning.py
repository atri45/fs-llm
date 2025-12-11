# evaluate_ptuning.py

import os
import torch
import json
import transformers
from transformers import GenerationConfig
from tqdm import tqdm
import logging

# --- 1. 从 FederatedScope 导入必要的模块 ---
# (确保你的项目在 PYTHONPATH 中，或者从项目根目录运行)
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl, get_tokenizer
from federatedscope.llm.model.model_builder import get_llm
try:
    from peft import PeftModel, set_peft_model_state_dict
except ImportError:
    raise ImportError("Please install `peft` library first (`pip install peft`).")

transformers.logging.set_verbosity(40)
logger = logging.getLogger(__name__)

# --- 2. 从 humaneval.py 复制辅助函数 ---
# (clean_answer, DEBUG, NUM_ANSWERS_PER_QUESTION 等)
DEBUG = False
NUM_ANSWERS_PER_QUESTION = 5
# ... (将 humaneval.py 中的 clean_answer 函数完整地复制到这里) ...
def clean_answer(code):
    """
    Borrow from: https://github.com/FSoft-AI4Code/CodeCapybara
    """
    def pad_spaces(s, num=4):
        n = 0
        while n < len(s) and s[n] == " ":
            n += 1
        if n != num:
            s = " " * num + s[n:]
        return s

    # 1. remove the special char \u00a0
    code = code.replace('\u00a0', '')
    # # 2. remove everything after "\n\n"
    # code = code.split("\n\n")[0]
    # 3. remove everything after the following stop sequences
    # Reference: https://github.com/openai/human-eval
    for stop_seq in ['\nclass', '\ndef', '\n#', '\nif', '\nprint', '\nassert']:
        code = code.split(stop_seq)[0]
    # 4. pad to four space to avoid `unindent` error
    code = pad_spaces(code, 4)
    return code


# --- 3. 主评估函数 ---
@torch.no_grad()
def main():
    # --- a. 加载配置 ---
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)
    
    device = f'cuda:{init_cfg.device}' if init_cfg.device > -1 else 'cpu'

    # --- b. 【核心】健壮的模型加载流程 ---
    logger.info("--- Starting Robust Model Loading ---")
    
    # i. 加载 Tokenizer
    model_name, model_hub = init_cfg.model.type.split('@')
    tokenizer, _ = get_tokenizer(model_name, init_cfg.data.root,
                                 init_cfg.llm.tok_len, model_hub)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ii. 使用 get_llm 加载带【随机初始化PEFT结构】的模型
    #    get_llm 会根据你的 yaml (finetune_type: p_tuning) 返回一个 PeftModel
    logger.info("Loading base model and attaching random PEFT adapter...")
    model = get_llm(init_cfg)

    # iii. 手动加载你训练好的【PEFT权重】
    ckpt_path = init_cfg.federate.save_to
    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading fine-tuned PEFT weights from: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            adapter_state_dict = ckpt.get('model', ckpt)

            # 找到 PeftModel 实例
            peft_model_instance = model.model if hasattr(model, 'model') else model
            
            if isinstance(peft_model_instance, PeftModel):
                # 检查是否是 P-Tuning 的 state_dict (通过检查特征键名)
                                # 1. 从 peft_config 字典中获取默认的配置对象
                #    通常适配器的名字是 'default'
                peft_config_obj = peft_model_instance.peft_config.get('default')

                if peft_config_obj is None:
                    raise ValueError("Could not find a 'default' PEFT config in the model.")

                # 2. 检查 state_dict 和配置类型是否匹配
                is_ptuning_dict = any('prompt_encoder' in k for k in adapter_state_dict.keys())

                if is_ptuning_dict and hasattr(peft_config_obj, 'peft_type') and peft_config_obj.peft_type == "P_TUNING":
                    logger.info("P-Tuning state_dict and model config detected. Loading weights directly into PromptEncoder.")
                    
                    # --- START OF FIX ---
                    # 1. 使用正确的属性来获取 PromptEncoder
                    if hasattr(peft_model_instance, 'prompt_encoder'):
                        prompt_encoder = peft_model_instance.prompt_encoder['default'] # 通常在 'default' 键下
                    else:
                        raise AttributeError("Could not find 'prompt_encoder' attribute on the PeftModel.")
                    
                    # 2. 准备要加载的 state_dict
                    #    load_state_dict 期望的键名不带 'prompt_encoder.default.' 前缀
                    state_dict_for_encoder = {}
                    prefix_to_strip = 'prompt_encoder.default.'
                    for key, value in adapter_state_dict.items():
                        if key.startswith(prefix_to_strip):
                            new_key = key[len(prefix_to_strip):]
                            state_dict_for_encoder[new_key] = value
                    
                    # 3. 加载权重
                    if state_dict_for_encoder:
                        prompt_encoder.load_state_dict(state_dict_for_encoder)
                        logger.info("Successfully loaded PEFT weights for P-Tuning.")
                    else:
                        logger.warning("Could not find any weights with 'prompt_encoder.default.' prefix in the checkpoint.")

                else:
                    # 对于 LoRA 或其他情况，继续使用 set_peft_model_state_dict
                    logger.info(f"'{getattr(peft_config_obj, 'peft_type', 'Unknown')}' PEFT type detected. Using `set_peft_model_state_dict`.")
                    set_peft_model_state_dict(peft_model_instance, adapter_state_dict)
                    logger.info("Successfully loaded PEFT weights.")
            else:
                logger.warning("Model is not a PeftModel. Attempting standard load_state_dict.")
                model.load_state_dict(adapter_state_dict, strict=False)

        except Exception as e:
            logger.error(f"Failed to load PEFT weights. Evaluation will use the raw model. Error: {e}", exc_info=True)
    else:
        logger.warning(f"Checkpoint path '{ckpt_path}' not specified or not found. Evaluation will use the raw model.")

    # iv. 准备模型用于推理
    if init_cfg.train.is_enable_half:
        model.half()
    model.to(device)
    model.eval()

    logger.info("--- Model loading complete ---")
    
    # --- c. 加载 HumanEval 数据集 ---
    # (与 humaneval.py 的逻辑相同)
    out_file = f'{init_cfg.federate.save_to}_humaneval_answer.jsonl'
    fp = os.path.join(init_cfg.data.root, 'HumanEval.jsonl.gz')
    if not os.path.exists(fp):
        download_url(
            'https://github.com/openai/human-eval/raw/'
            '463c980b59e818ace59f6f9803cd92c749ceae61/'
            'data/HumanEval.jsonl.gz', init_cfg.data.root)
    list_data_dict = load_jsonl(fp,
                                instruction='prompt',
                                input='entry_point',
                                category='task_id',
                                output='test',
                                is_gzip=True)

    # --- d. 执行生成和评估 ---
    answers = []
    for sample in tqdm(list_data_dict):
        input_text = sample['instruction']
        generation_config = GenerationConfig(
            temperature=0.1,
            top_k=40,
            top_p=0.75,
            do_sample=True,
            num_return_sequences=NUM_ANSWERS_PER_QUESTION,
        )
        generate_kwargs = dict(
            generation_config=generation_config,
            max_new_tokens=128,
        )
        
        try:
            # 直接调用 model.generate
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            output_ids = model.generate(input_ids=input_ids, **generate_kwargs)
            
            # 解码
            # (这里的解码逻辑比 FSChatBot.generate 更简单直接)
            completions_text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
            model_completions = completions_text

        except torch.cuda.OutOfMemoryError as error:
            print(error)
            model_completions = ['' for _ in range(NUM_ANSWERS_PER_QUESTION)]

        for i, completion in enumerate(model_completions):
            completion = clean_answer(completion)
            answers.append(dict(task_id=sample['category'], completion=completion))

    # --- e. 保存结果 ---
    with open(out_file, 'w') as f:
        for answer in answers:
            f.write(json.dumps(answer) + '\n')
            
    logger.info(f"HumanEval predictions saved to: {out_file}")

if __name__ == "__main__":
    main()