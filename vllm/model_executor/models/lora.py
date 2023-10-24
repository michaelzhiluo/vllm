"""
References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""
import math

import torch
from torch import nn


SUPPORTED_LORA_LAYERS = [
    'ColumnParallelLinear',
    'RowParallelLinear',
]

def apply_lora_wrapper(model, lora_config):
    named_module_tuple = list(model.named_modules())
    for name, module in named_module_tuple:
        module_name = type(module).__name__
        if module_name in SUPPORTED_LORA_LAYERS:
            wrapper = BatchedLoraWrapper(module, **lora_config)
            path = name.split('.')
            parent_module = model
            for sub_name in path[:-1]:  # Traverse down to the parent of the target module
                parent_module = getattr(parent_module, sub_name)
            setattr(parent_module, path[-1], wrapper)


class BatchedLoraWrapper(nn.Module):
    def __init__(self, layer, lora_num_models=1, lora_rank=4, lora_alpha=1):
        super().__init__()
        # Replicate layer state to this LayerWrapper.
        self.__dict__.update(layer.__dict__)

        # Get the input and output size of the layer for LORA.
        self._get_fan_in_out(layer)
        self.ffn = layer.forward

        self.lora_A = nn.Parameter(torch.zeros((lora_num_models, self.lora_input_size, lora_rank), device=torch.cuda.current_device(),))
        self.lora_B = nn.Parameter(torch.zeros((lora_num_models, self.lora_output_size, lora_rank), device=torch.cuda.current_device(),))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

        self.num_lora_models = lora_num_models
        self.lora_alpha, self.lora_rank = lora_alpha, lora_rank
        self.lora_scaling = lora_alpha / lora_rank
        self.lora_enabled = True

    def _get_fan_in_out(self, layer):
        """
        Automatically determine the input and outsize of the layer.
        """
        if hasattr(self, 'input_size'):
            self.lora_input_size = layer.input_size
        elif hasattr(self, 'in_features'):
            self.lora_input_size = layer.in_features
        else:
            raise ValueError(f'Cannot find input size of layer: {layer}')

        if hasattr(self, 'output_size'):
            self.lora_output_size = layer.output_size
        elif hasattr(self, 'out_features'):
            self.lora_output_size = layer.out_features
        else:
            raise ValueError(f'Cannot find output size of layer: {layer}')

    
    def forward(self, *args, **kwargs):
        # Assume args[0] is the input, X.
        X = args[0]

        # Layer is expected to be a FC layer.
        outputs = self.ffn(*args, **kwargs)
        if not self.lora_enabled:
            return outputs

        if isinstance(outputs, tuple) or isinstance(outputs, list):
            output = outputs[0]
            extra_outputs = outputs[1:]
        else:
            output = outputs
        
        # NWR - Num Model x Width x Rank, BW - Batch x Width
        inter_tensor = torch.einsum('NWR,BW->NBR', self.lora_A, X)
        lora_output = output + torch.einsum('NBR,NWR->BW', inter_tensor, self.lora_B) * self.lora_scaling
        return lora_output, extra_outputs

    def disable_lora(self):
        self.lora_enabled=False

    def enable_lora(self):
        self.lora_enabled = True