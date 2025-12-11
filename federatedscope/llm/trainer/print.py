import torch
from transformers import AutoModelForCausalLM

# --- 在这里修改成你的模型所在的本地路径 ---
model_path = "/home/zzh/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Loading model from local path: {model_path}\n")

try:
    # 使用 AutoModelForCausalLM 从本地路径加载
    # trust_remote_code=True 依然是必需的，因为它需要加载该模型目录下的自定义Python代码文件
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        # 如果你的GPU显存有限，可以加上 device_map='auto' 让它自动分配层到CPU和GPU
        # device_map='auto'
    )
    
    print("\n" + "="*20 + " Model Structure " + "="*20)
    # 打印完整的模型结构，你可以从中找到 transformer_block_class_name
    print(model)
    
    print("\n" + "="*20 + " All Module Names " + "="*20)
    # 打印模型中所有模块的名称，你可以从中找到 target_modules
    found_modules = set()
    for name, module in model.named_modules():
        # 我们只关心最后一级的模块名，以避免层级前缀的干扰
        module_name_part = name.split('.')[-1]
        
        # 筛选出可能是线性层的模块名
        # 这是一个常见的启发式规则，你可以根据需要调整
        if "proj" in module_name_part or "fc" in module_name_part or "dense" in module_name_part:
            if isinstance(module, torch.nn.Linear):
                found_modules.add(module_name_part)
        
        # 打印所有名称，以供手动检查
        # print(name) 

    if found_modules:
        print("\nFound potential target_modules for LoRA:")
        # 转换成列表并排序，方便复制
        sorted_modules = sorted(list(found_modules))
        print(sorted_modules)
        print("\nRecommendation for target_modules in your YAML file:")
        print(f"target_modules: {sorted_modules}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check if the path is correct and contains all necessary model files (config.json, pytorch_model.bin, etc.).")