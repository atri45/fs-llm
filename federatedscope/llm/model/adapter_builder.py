import os
import torch
import torch.nn as nn
from collections import OrderedDict
import logging

try:
    from peft import PeftModel, get_peft_model_state_dict
except ImportError:
    class PeftModel: pass
    get_peft_model_state_dict = None
try:
    import adapters
except ImportError:
    adapters = None

logger = logging.getLogger(__name__)

def enable_adapter(model, package, adapter, **kwargs):
    """
    Enables an adapter for a given model and package.

    Args:
        model: A pre-trained model from HuggingFace Transformers library.
        package: A string indicating the name of the package that provides
            the adapter. Currently, only 'peft' and 'adapterhub' is supported.
        adapter: A string indicating the name of the adapter to enable. The
            available adapters depend on the package.
        **kwargs: Additional keyword arguments that are passed to the
            adapter configuration.

    Returns:
        A model object that has the adapter enabled.

    Raises:
        NotImplementedError: If the package or the adapter is not supported.
    """
    adapter = adapter.lower()
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        Support methods:
            LoRA
            Prefix Tuning
            P-Tuning
            Prompt Tuning
            AdaLoRA
        """
        from peft import get_peft_model, TaskType
        if adapter == 'lora':
            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prefix':
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prompt':
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'p-tuning':
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(
                peft_type="P_TUNING",
                task_type=TaskType.CAUSAL_LM,  # Qwen是Causal LM (仅解码器)
                
                # --- 核心超参数 (你可以调整) ---
                num_virtual_tokens=32,          # 虚拟提示的长度。20-100是常用范围。32是一个好的起点。
                
                # --- 必须与Qwen-1.5B模型架构匹配的参数 ---
                token_dim=1536,                 # 模型的隐藏层维度 (hidden_size)
                num_transformer_submodules=1,   # 对于仅解码器模型，总是1
                num_attention_heads=12,         # Qwen-1.5B的注意力头数 (请再次确认)
                num_layers=28,                  # Qwen-1.5B的层数
                
                # --- Prompt Encoder 的内部结构配置 ---
                encoder_reparameterization_type="MLP", # 使用MLP来生成提示嵌入，比LSTM更常用
                encoder_hidden_size=1024        # Prompt Encoder内部MLP的隐藏层大小。通常是token_dim的一半左右。
            )
            model = get_peft_model(model, peft_config)
        elif adapter == 'adalora':
            from peft import AdaLoraConfig
            peft_config = AdaLoraConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'loha':
            from peft import LoHaConfig
            peft_config = LoHaConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'lokr':
            from peft import LoKrConfig
            peft_config = LoKrConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'xlora':
            from peft import XLoraConfig
            peft_config = XLoraConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'vera':
            from peft import VeraConfig
            peft_config = VeraConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'vblora':
            from peft import VBLoRAConfig
            peft_config = VBLoRAConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'ft':
            from peft import FourierFTConfig
            peft_config = FourierFTConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'oft':
            from peft import OFTConfig
            peft_config = OFTConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'boft':
            from peft import BOFTConfig
            peft_config = BOFTConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'ia3':
            from peft import IA3Config
            peft_config = IA3Config(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError
        model.print_trainable_parameters()

    elif package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        Support methods:
            Bottleneck Adapters
            Prefix Tuning
            LoRA
            Compacter
            Adapter Fusion
            Invertible Adapters
            Parallel block
        """
        # TODO:  After supporting adapterhub, we will move the following
        #   parameters in yaml file for users' convenient
        import adapters
        adapters.init(model)

        if adapter == 'lora':
            from adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=32)
            model.add_adapter("lora", config=config)
            model.train_adapter(['lora'])

        elif adapter == 'bottleneck':
            from adapters import BnConfig

            config = BnConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")

            # from adapters import AdapterPlusConfig
            # config = AdapterPlusConfig()
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])

        elif adapter == 'language':
            from adapters import SeqBnInvConfig

            config = SeqBnInvConfig()
            model.add_adapter("language_adapter", config=config)
            model.train_adapter(['language_adapter'])

        elif adapter == 'prefix':
            from adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])

        elif adapter == 'compacter':
            from adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("compacter", config=config)
            model.train_adapter(['compacter'])

        elif adapter == 'ia3':
            from adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3", config=config)
            model.train_adapter(['ia3'])

        # exist bug
        elif adapter == 'vera':
            from adapters import VeraConfig

            config = VeraConfig()
            model.add_adapter("vera_config", config=config)
            model.train_adapter(['vera_config'])

        elif adapter == 'prompt':
            from adapters import PromptTuningConfig

            config = PromptTuningConfig(prompt_length=10)
            model.add_adapter("prompt_tuning", config=config)
            model.train_adapter(['prompt_tuning'])

        elif adapter == 'loreft':
            from adapters import LoReftConfig

            config = LoReftConfig()
            model.add_adapter("loreft", config=config)
            model.train_adapter(['loreft'])

        elif adapter == 'noreft':
            from adapters import NoReftConfig

            config = NoReftConfig()
            model.add_adapter("noreft", config=config)
            model.train_adapter(['noreft'])

        elif adapter == 'direft':
            from adapters import DiReftConfig

            config = DiReftConfig()
            model.add_adapter("direft", config=config)
            model.train_adapter(['direft'])

        elif adapter == 'mam':
            from adapters import MAMConfig

            config = MAMConfig()
            model.add_adapter("mam_adapter", config=config)
            model.train_adapter(['mam_adapter'])

        elif adapter == 'unipelt':
            from adapters import UniPELTConfig

            config = UniPELTConfig()
            model.add_adapter("unipelt", config=config)
            model.train_adapter(['unipelt'])

        elif adapter == 'union':
            from adapters import AdapterConfig, ConfigUnion

            # TODO: configure these args in cfg
            config = ConfigUnion(
                AdapterConfig(mh_adapter=True,
                              output_adapter=False,
                              reduction_factor=16,
                              non_linearity="relu"),
                AdapterConfig(mh_adapter=False,
                              output_adapter=True,
                              reduction_factor=2,
                              non_linearity="relu"),
            )
            model.add_adapter("union_adapter", config=config)
            model.train_adapter(['union_adapter'])


        else:
            raise NameError(
                f"There is no adapter named {adapter} in {package}")
    else:
        raise NotImplementedError
    return model


class AdapterModel(nn.Module):
    """
    A wrapper class for a model that can use adapters for fine-tuning.

    This class inherits from torch.nn.Module and implements a wrapper for a
    model that can optionally use adapters for fine-tuning. Adapters are small
    modules that can be inserted between the layers of a pretrained model and
    trained on a specific task, while keeping the original parameters frozen.
    This class can use different adapter packages and methods, such as PEFT
    and LoRA. It also provides methods for saving and loading the model state
    dict, as well as generating text using the model.

    Attributes:
        model: A torch.nn.Module object that represents the original or
            adapted model.

    """
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        """
        Initializes the wrapper with the given model and arguments.

        Args:
            model: A torch.nn.Module object that represents the original model.
            use_adapter: A boolean indicating whether to use adapters for
                fine-tuning. Default is False.
            *args: Additional positional arguments to pass to the adapter
                package or method.
            **kwargs: Additional keyword arguments to pass to the adapter
                package or method. These may include adapter_package,
                adapter_method, etc.
        """
        super().__init__()

        self.model = None
        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')

            self.model = enable_adapter(model, adapter_package, adapter_method,
                                        **kwargs)
        else:
            self.model = model

    def forward(self, *args, **kwargs):
        """
        Calls the forward method of the wrapped model.

        Args:
            *args: Positional arguments to pass to the model's forward method.
            **kwargs: Keyword arguments to pass to the model's forward method.

        Returns:
            The output of the model's forward method.
        """
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Calls the generate method of the wrapped model.

        Args:
            *args: Positional arguments to pass to the model's generate method.
            **kwargs: Keyword arguments to pass to the model's generate method.

        Returns:
            The output of the model's generate method.
        """
        try:
            res = self.model.generate(*args, **kwargs)
        except RuntimeError as e:
            # When does evaluation in HELM,
            # half precision will cause RuntimeError,
            # the following solves it
            if 'do_sample' in kwargs.keys():
                del kwargs['do_sample']
                res = self.model.generate(*args, **kwargs)
            else:
                raise RuntimeError(e)
        return res

    def state_dict(self, return_trainable=True, *args, **kwargs):
        """
        Returns the state dict of the wrapped model.

        Args:
            return_trainable: A boolean indicating whether to return only the
                trainable parameters of the model. Default is True.
            *args: Additional positional arguments to pass to the model's
                state_dict method.
            **kwargs: Additional keyword arguments to pass to the model's
                state_dict method.

        Returns:
            A dictionary containing the state dict of the model. If
            return_trainable is True, only the parameters that require grad are
            included. Otherwise, all parameters are included.
        """
        if return_trainable:
            return self.get_trainable_state_dict()
        else:
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        """
        Loads the state dict into the wrapped model.

        Args:
            state_dict: A dictionary containing the state dict to load into
                the model.
            strict: A boolean indicating whether to strictly enforce that the
                keys in state_dict match the keys returned by this module’s
                state_dict() function. Default is False.
        """
        return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        """
        Returns only the trainable parameters of the wrapped model.

        This method can be used to get only the parameters that require grad,
        such as adapters or task-specific layers.

        Returns:
            A dictionary containing the state dict of the trainable parameters
            of the model.
        """
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict

    def save_model(self, path, state=0):
        """
        Saves the model state dict and the current round to a file.

        Args:
            path: A string representing the file path to save the model to.
            state: An integer representing the current round of training or
                evaluation. Default is 0.

        """
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    # TODO: Fix `__getattr__`
    # def __getattr__(self, item):
    #     return getattr(self.model, item)
