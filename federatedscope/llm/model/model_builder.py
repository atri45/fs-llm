import torch
from federatedscope.llm.model.adapter_builder import AdapterModel
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
 # <-- 确保导入 AutoConfig
import logging

logger = logging.getLogger(__name__)

def get_model_from_huggingface(model_name, config):
    """
    Load a causal language model from HuggingFace transformers library.

    Args:
        model_name (str): The name of the pre-trained model to load.
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        AutoModelForCausalLM: A causal language model object.
    """
    # 准备传递给 from_pretrained 的通用参数
    kwargs = {}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model
    
    # 很多新模型需要信任远程代码
    if config.model.get("trust_remote_code", False):
        kwargs['trust_remote_code'] = True
        
    # 处理半精度
    if config.train.is_enable_half:
        kwargs['torch_dtype'] = torch.float16


    logger.info("BitsAndBytes quantization is enabled.")
    # 从配置中动态创建 BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16 # 使用 bfloat16 进行计算以保持精度
    )
    kwargs['quantization_config'] = bnb_config

    # --- START OF FIX for Activation Checkpointing ---
    # 1. 首先加载模型配置
    try:
        loaded_config = AutoConfig.from_pretrained(model_name, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load model config for '{model_name}'. "
                     f"Please check model name and cache path. Error: {e}")
        raise e

    # 2. 修改配置：在使用激活检查点时，必须禁用 use_cache
    if config.federate.use_activation_checkpointing:
        if hasattr(loaded_config, 'use_cache'):
            logger.info("Activation checkpointing is enabled. Setting `use_cache=False` in model config.")
            loaded_config.use_cache = False
        else:
            logger.warning(f"Attempted to set `use_cache=False` for activation checkpointing, "
                           f"but the model config for '{model_name}' does not have a 'use_cache' attribute.")
            
    # 3. 将修改后的配置对象添加到 kwargs 中，以便传递给模型加载函数
    kwargs['config'] = loaded_config
    # --- END OF FIX ---

    # 4. 使用最终的参数集来加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load model weights for '{model_name}'. "
                     f"Please check model name and configurations. Error: {e}")
        raise e

    return model


def get_model_from_modelscope(model_name, config):
    """
    Load a causal language model from ModelScope models library.

    Args:
        model_name (str): The name of the pre-trained model to load.
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        Model: A causal language model object.
    """
    from modelscope import AutoModelForCausalLM

    kwargs = {}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config):
    """
    Get a causal language model based on the configuration.

    Args:
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        AdapterModel: A causal language model object with optional adapter
            layers.
    """
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name, config=config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # Resize LLM model based on settings
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len,
                      model_hub)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)

    return model
