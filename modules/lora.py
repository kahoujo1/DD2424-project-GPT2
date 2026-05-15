"""
Implementation of LoRA (Low-Rank Adaptation) linear layer.
"""

import math

import torch
from torch import nn

from models.gpt2 import GPT2Model


VALID_TARGET_MODULES = ['query', 'key', 'value', 'dense']

class LoRALinear(nn.Linear):
    """
    LoRA linear layer that adds low-rank adaptation to a standard linear layer.
    """
    def __init__(self, in_features, out_features, r: int = 4, alpha:float = 1.0):
        super().__init__(in_features, out_features)
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        # Initialize LoRA matrices
        self.init_lora_params()
        # make sure the original weights are not updated during training
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def init_lora_params(self):
        """
        Initializes the LoRA matrices.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LoRA linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # base transformation
        output = super().forward(x)
        # LoRA adaptation
        lora_output = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return output + lora_output

def exchange_model_layers(model: GPT2Model, r = 4, alpha = 1.0, target_modules = ['query', 'value']) -> GPT2Model:
    """
    Exchanges the linear layers in the Multi-Head Attention modules of the GPT2Model with LoRA linear layers.

    Args:
        model (GPT2Model): The GPT2Model to modify.
        r (int): The rank for the LoRA adaptation.
        alpha (float): The scaling factor for the LoRA adaptation.
        target_modules (list of str): The names of the linear layers to replace with LoRA layers.
    
    Returns:
        GPT2Model: The modified GPT2Model with LoRA linear layers.
    """
    print(f'Initializing Lora, params: alpha={alpha}, r={r}, target_modules={target_modules}')
    # firstly, freeze all original parameters
    for param in model.parameters():
        param.requires_grad = False
    # check if the target modules are valid
    if not all(target in VALID_TARGET_MODULES for target in target_modules):
        raise ValueError(f"Invalid target modules. Valid options are: {VALID_TARGET_MODULES}")
    for i in range(len(model.gpt_layers)):
        gpt_layer = model.gpt_layers[i]
        for target in target_modules:
            if target == 'dense': # the W_o layer after attention
                linear_layer = gpt_layer.attention_dense
            else: # the query, key, value layers in the attention module    
                linear_layer = getattr(gpt_layer.self_attention, target)
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            lora_linear = LoRALinear(in_features, out_features, r, alpha)
            # Match the device and dtype of the original layer
            lora_linear = lora_linear.to(device=linear_layer.weight.device, 
                                         dtype=linear_layer.weight.dtype)
            # Copy the original weights to the LoRA linear layer (for inference).
            with torch.no_grad():
                lora_linear.weight.copy_(linear_layer.weight)
                if linear_layer.bias is not None:
                    lora_linear.bias.copy_(linear_layer.bias)
            if target == 'dense':
                gpt_layer.attention_dense = lora_linear
            else:
                setattr(gpt_layer.self_attention, target, lora_linear)

    # Quick sanity check for trainable parameters
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(f"Training: {name}") # Uncomment to see exactly what is un-frozen

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}%"
    )
    return model
            
def save_lora_weights(model: GPT2Model, filepath: str):
    """
    Saves the LoRA weights of the modified GPT2Model.

    Args:
        model (GPT2Model): The modified GPT2Model with LoRA linear layers.
        filepath (str): The path to save the LoRA weights.
    """
    lora_state_dict = {
        name: param.data for name, param in model.named_parameters() if param.requires_grad
    }
    torch.save(lora_state_dict, filepath)
    print(f"LoRA weights saved to {filepath}")

def load_lora_weights(model: GPT2Model, filepath: str):
    """
    Loads the LoRA weights into the modified GPT2Model.

    Args:
        model (GPT2Model): The modified GPT2Model with LoRA linear layers.
        filepath (str): The path to load the LoRA weights from.
    """
    lora_state_dict = torch.load(filepath)
    model.load_state_dict(lora_state_dict, strict=False) # strict=False to allow loading only the LoRA weights
    print(f"LoRA weights loaded from {filepath}")