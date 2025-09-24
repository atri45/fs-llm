import random
import numpy as np

from federatedscope.core.splitters import BaseSplitter


class MetaSplitter(BaseSplitter):
    """
    This splitter split dataset with meta information with LLM dataset.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num, **kwargs):
        super(MetaSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            # 确保你的数据字典里有 'categories' 这个键
            if 'categories' not in tmp_dataset[0]:
                raise ValueError("MetaSplitter requires a 'categories' key in the data items.")
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')

        categories = sorted(list(set(label))) # 使用 sorted 保证每次运行类别顺序一致
        idx_slice = []
        for cat in categories:
            idx_slice.append(np.where(np.array(label) == cat)[0].tolist())
        
        # 为了调试，可以取消下面的注释看看每个类别的数量
        # print(f"MetaSplitter found {len(categories)} categories with sizes: {[len(s) for s in idx_slice]}")

        # random.shuffle(idx_slice) # 打乱类别分配的顺序

        # Merge to client_num pieces
        new_idx_slice = [[] for _ in range(self.client_num)] # 正确的初始化方式
        for i in range(len(categories)):
            new_idx_slice[i % self.client_num].extend(idx_slice[i]) # 使用 extend 合并列表

        # --- FIX IS HERE ---
        # 使用修正后的 new_idx_slice 来创建 data_list
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in new_idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in new_idx_slice]
        
        # 添加一个内部的健全性检查
        assert sum([len(d) for d in data_list]) == len(dataset)

        return data_list
