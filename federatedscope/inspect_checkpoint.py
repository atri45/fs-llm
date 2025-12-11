# inspect_checkpoint.py

import torch
import argparse
import os

def inspect_checkpoint(file_path, top_k=20, filter_keyword=None):
    """
    加载一个 PyTorch 检查点文件 (.ckpt) 并打印其内部结构。

    Args:
        file_path (str): .ckpt 文件的路径。
        top_k (int): 显示 state_dict 中前 k 个键名。
        filter_keyword (str, optional): 只显示包含此关键字的键名。
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"--- Inspecting Checkpoint: {file_path} ---")

    try:
        # 加载到 CPU 以避免占用 GPU 显存
        ckpt = torch.load(file_path, map_location=torch.device('cpu'))

        # 1. 检查顶层结构
        print("\n[1] Top-level keys in the checkpoint file:")
        if isinstance(ckpt, dict):
            print(list(ckpt.keys()))
        else:
            print(f"Checkpoint is not a dictionary. It is a a {type(ckpt)}.")
            # 如果不是字典，可能直接是一个 tensor 或其他对象
            if isinstance(ckpt, torch.Tensor):
                print(f"  - Tensor shape: {ckpt.shape}")
                print(f"  - Tensor dtype: {ckpt.dtype}")
            return # 后续步骤依赖于字典结构

        # 2. 深入检查 'model' state_dict
        if 'model' in ckpt:
            state_dict = ckpt['model']
            print("\n[2] Found a 'model' key. Inspecting its state_dict:")
        else:
            # 假设整个 ckpt 文件就是一个 state_dict
            state_dict = ckpt
            print("\n[2] No 'model' key found. Assuming the entire file is a state_dict:")

        if not isinstance(state_dict, dict):
            print(f"  - Error: The 'model' value is not a dictionary, but a {type(state_dict)}.")
            return
            
        all_keys = list(state_dict.keys())
        total_params = len(all_keys)
        print(f"  - Total number of parameters/buffers in state_dict: {total_params}")

        # 3. 筛选并打印键名
        keys_to_show = all_keys
        if filter_keyword:
            keys_to_show = [k for k in all_keys if filter_keyword in k]
            print(f"\n[3] Showing {len(keys_to_show)} keys (out of {total_params}) that contain '{filter_keyword}':")
        else:
            print(f"\n[3] Showing the first {min(top_k, total_params)} keys:")
        
        for i, key in enumerate(keys_to_show):
            if i >= top_k and not filter_keyword:
                print(f"  ... (and {total_params - top_k} more)")
                break
            
            tensor = state_dict[key]
            print(f"  - {key:<80} shape: {list(tensor.shape)}, dtype: {tensor.dtype}")

    except Exception as e:
        print(f"\nAn error occurred while reading the file: {e}")

    print("\n--- Inspection Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint file.")
    parser.add_argument("file_path", type=str, help="Path to the .ckpt file to inspect.")
    parser.add_argument("-k", "--top_k", type=int, default=20, help="Number of state_dict keys to display.")
    parser.add_argument("-f", "--filter", type=str, default=None, help="Only show keys containing this keyword.")
    
    args = parser.parse_args()
    
    inspect_checkpoint(args.file_path, args.top_k, args.filter)