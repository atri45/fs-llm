# scripts/create_model_cache.py
import os
import sys
import torch
import numpy as np
import logging
import re
from collections import OrderedDict

# 确保能找到 federatedscope 模块
# (根据你的项目结构调整路径)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.configs.config import global_cfg
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.core.auxiliaries.logging import update_logger

logger = logging.getLogger(__name__)

def _sanitize_model_name_for_path(model_name: str) -> str:
    base_name = os.path.basename(model_name)
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', base_name)

def main():
    # --- 1. 加载配置 ---
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)
    update_logger(init_cfg, clear_before_add=True)
    
    # --- 2. 加载模型 ---
    logger.info("Loading model...")
    # 确保配置正确，特别是 model.type 和 model.trust_remote_code
    model = get_llm(init_cfg)

    full_state_dict = model.state_dict(return_trainable=False)
    
    logger.info(f"Successfully extracted a total of {len(full_state_dict)} parameters.")
    
    # --- 3. 获取缓存路径和模型名 ---
    cache_path = init_cfg.federate.model_cache_path
    if not cache_path:
        raise ValueError("`federate.model_cache_path` must be specified in the config.")
        
    sanitized_model_name = _sanitize_model_name_for_path(init_cfg.model.type)
    model_specific_cache_path = os.path.join(cache_path, sanitized_model_name)
    
    # --- 4. 执行缓存创建逻辑 (从 FedSpeedEngine 移植过来) ---
    logger.info(f"Creating layer-wise cache for model '{sanitized_model_name}' at {model_specific_cache_path}...")
    os.makedirs(model_specific_cache_path, exist_ok=True)
    
    num_params_saved = 0
    for name, param_data_to_save in full_state_dict.items():
        # 这里的 key 就是 state_dict 的 key，不需要再做复杂的归一化
        param_filename = name + ".npy"
        full_path = os.path.join(model_specific_cache_path, param_filename)
        
        # 确保子目录存在 (例如 'layers.0.self_attn...')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        numpy_array = param_data_to_save.cpu().numpy()
        np.save(full_path, numpy_array)
        num_params_saved += 1
        
    # 创建完成标志
    done_file = os.path.join(model_specific_cache_path, ".creation_done")
    with open(done_file, 'w') as f: f.write('done')
    
    logger.info(f"Successfully created cache for {num_params_saved} parameters.")
    logger.info(f"You can now package the directory: '{model_specific_cache_path}' and distribute it to all clients.")

if __name__ == "__main__":
    main()