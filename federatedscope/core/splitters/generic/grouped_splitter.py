import numpy as np
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.generic import IIDSplitter

class GroupedSplitter(BaseSplitter):
    """
    This splitter first splits dataset into N groups in a Non-IID manner,
    and then splits the data of each group to clients within that group in an
    IID manner.

    Args:
        client_num (int): The total number of clients.
        num_groups (int): The number of groups to form.
        base_splitter (BaseSplitter): The splitter for Non-IID splitting
                                      among groups.
    """
    def __init__(self, client_num, num_groups, base_splitter):
        if client_num < num_groups:
            raise ValueError(f"Total number of clients ({client_num}) must be "
                             f"greater than or equal to the number of groups "
                             f"({num_groups}).")
        
        self.num_groups = num_groups
        self.base_splitter = base_splitter
        # We always use IID splitter for intra-group splitting
        self.intra_splitter = IIDSplitter 
        super(GroupedSplitter, self).__init__(client_num)

        # Assign clients to groups as evenly as possible
        # Example: 8 clients, 3 groups -> [3, 3, 2] clients per group
        print("[*] GroupedSplitter `__init__`: Initializing client groups...")
        client_indices = np.arange(1, client_num + 1) # Use 1-based indexing for clarity
        self.clients_per_group_indices = np.array_split(client_indices, num_groups)
        self.client_groups = [len(group) for group in self.clients_per_group_indices]
        print(f"    - Clients assigned to groups as follows (client counts): {self.client_groups}")
        for i, group_indices in enumerate(self.clients_per_group_indices):
            print(f"      - Group {i+1}: {len(group_indices)} clients (IDs: {list(group_indices)})")

    def __call__(self, dataset, **kwargs):
        from torch.utils.data import Dataset, Subset
        print("\n[*] GroupedSplitter `__call__`: Starting data splitting process...")
        print(f"    - Total dataset size: {len(dataset)}")
        
        # Step 1: Split the dataset into `num_groups` Non-IID chunks
        print(f"\n    [Step 1] Performing Non-IID split into {self.num_groups} groups using `{type(self.base_splitter).__name__}`...")
        # Step 1: Split the dataset into `num_groups` Non-IID chunks
        # We use the provided base_splitter for this.
        # Temporarily set the client_num of base_splitter to num_groups
        original_client_num = self.base_splitter.client_num
        self.base_splitter.client_num = self.num_groups
        
        group_data = self.base_splitter(dataset, **kwargs)
        
        # Restore the original client_num
        self.base_splitter.client_num = original_client_num

        group_data_sizes = [len(d) for d in group_data]
        print(f"    - Non-IID split resulted in groups with data sizes: {group_data_sizes}")
        assert sum(group_data_sizes) == len(dataset), "Data loss after inter-group split!"

        # Step 2: For each group's data, split it IID among the clients
        print(f"\n    [Step 2] Performing IID split within each group for {self.client_num} total clients...")

        # Step 2: For each group's data, split it IID among the clients
        # in that group.
        client_data_list = []
        for i, g_data in enumerate(group_data):
            num_clients_in_group = self.client_groups[i]
            group_client_ids = self.clients_per_group_indices[i]
            print(f"      - Processing Group {i+1} (data size: {len(g_data)}) for {num_clients_in_group} clients (IDs: {list(group_client_ids)})...")
            if num_clients_in_group > 0:
                # Instantiate an IID splitter for the clients in this group
                iid_splitter = self.intra_splitter(num_clients_in_group)
                # Split the group's data IID
                intra_group_split = iid_splitter(g_data)
                
                intra_group_sizes = [len(d) for d in intra_group_split]
                print(f"        - IID split within this group resulted in data sizes: {intra_group_sizes}")
                assert sum(intra_group_sizes) == len(g_data), f"Data loss during intra-group split for group {i+1}!"
                client_data_list.extend(intra_group_split)
        # Sanity check
        print("\n[*] GroupedSplitter `__call__`: Splitting process finished.")
        final_client_data_sizes = [len(d) for d in client_data_list]
        print(f"    - Final data sizes for each client: {final_client_data_sizes}")
        print(f"    - Total number of clients with data: {len(final_client_data_sizes)}")
        assert len(client_data_list) == self.client_num, \
            f"The number of final data splits ({len(client_data_list)}) does not match the total number of clients ({self.client_num})."
        print("[*] Sanity checks passed. Grouped splitting was successful.")
            
        return client_data_list