import logging

import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.splitter import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.splitter`, some modules are not '
        f'available.')


def get_splitter(config):
    """
    This function is to build splitter to generate simulated federation \
    datasets from non-FL dataset.

    Args:
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        An instance of splitter (see ``core.splitters`` for details)

    Note:
      The key-value pairs of ``cfg.data.splitter`` and domain:
        ===================  ================================================
        Splitter type        Domain
        ===================  ================================================
        lda	                 Generic
        iid                  Generic
        louvain	             Graph (node-level)
        random	             Graph (node-level)
        rel_type	         Graph (link-level)
        scaffold	         Molecular
        scaffold_lda       	 Molecular
        rand_chunk	         Graph (graph-level)
        ===================  ================================================
    """
    client_num = config.federate.client_num
    if config.data.splitter_args:
        kwargs = config.data.splitter_args[0]
    else:
        kwargs = {}

    for func in register.splitter_dict.values():
        splitter = func(config.data.splitter, client_num, **kwargs)
        if splitter is not None:
            return splitter
    # Delay import
    # generic splitter
    if config.data.splitter == 'grouped':
        print("\n" + "="*20 + " Building GroupedSplitter " + "="*20)
        from federatedscope.core.splitters.generic.grouped_splitter import GroupedSplitter
        from federatedscope.core.splitters.generic import LDASplitter, MetaSplitter
        
        # Check for required arguments
        if 'num_groups' not in kwargs:
            raise ValueError("`num_groups` must be specified in "
                             "`data.splitter_args` for grouped splitter.")
        if 'base_splitter_type' not in kwargs:
            raise ValueError("`base_splitter_type` (e.g., 'lda' or 'meta') "
                             "must be specified in `data.splitter_args`.")
        
        num_groups = kwargs.pop('num_groups')
        base_splitter_type = kwargs.pop('base_splitter_type')
        
        print(f"[*] GroupedSplitter: Found configuration:")
        print(f"    - Total clients: {client_num}")
        print(f"    - Number of groups: {num_groups}")
        print(f"    - Base splitter for inter-group split: '{base_splitter_type}'")

        # Create the base splitter instance based on the type
        if base_splitter_type == 'lda':
            print(f"    - Args for LDASplitter: {kwargs}")
            # Pass remaining kwargs (like alpha) to the base splitter
            base_splitter_instance = LDASplitter(client_num, **kwargs)
        elif base_splitter_type == 'meta':
            print(f"    - Args for MetaSplitter: {kwargs}")
            base_splitter_instance = MetaSplitter(client_num, **kwargs)
        else:
            raise ValueError(f"Unsupported base_splitter_type: "
                             f"{base_splitter_type}")
            
        splitter = GroupedSplitter(client_num, num_groups, base_splitter_instance)
        print("="*24 + " GroupedSplitter Built " + "="*25 + "\n")
    elif config.data.splitter == 'lda':
        from federatedscope.core.splitters.generic import LDASplitter
        splitter = LDASplitter(client_num, **kwargs)
    # graph splitter
    elif config.data.splitter == 'louvain':
        from federatedscope.core.splitters.graph import LouvainSplitter
        splitter = LouvainSplitter(client_num, **kwargs)
    elif config.data.splitter == 'random':
        from federatedscope.core.splitters.graph import RandomSplitter
        splitter = RandomSplitter(client_num, **kwargs)
    elif config.data.splitter == 'rel_type':
        from federatedscope.core.splitters.graph import RelTypeSplitter
        splitter = RelTypeSplitter(client_num, **kwargs)
    elif config.data.splitter == 'scaffold':
        from federatedscope.core.splitters.graph import ScaffoldSplitter
        splitter = ScaffoldSplitter(client_num, **kwargs)
    elif config.data.splitter == 'scaffold_lda':
        from federatedscope.core.splitters.graph import ScaffoldLdaSplitter
        splitter = ScaffoldLdaSplitter(client_num, **kwargs)
    elif config.data.splitter == 'rand_chunk':
        from federatedscope.core.splitters.graph import RandChunkSplitter
        splitter = RandChunkSplitter(client_num, **kwargs)
    elif config.data.splitter == 'iid':
        from federatedscope.core.splitters.generic import IIDSplitter
        splitter = IIDSplitter(client_num)
    elif config.data.splitter == 'meta':
        from federatedscope.core.splitters.generic import MetaSplitter
        splitter = MetaSplitter(client_num)
    else:
        logger.warning(f'Splitter {config.data.splitter} not found or not '
                       f'used.')
        splitter = None
    return splitter
