from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  #NOTE: We implemented this
  def add(self, input, output, dense_layer, dropout):
    """
    A helper function in the GPT-2 layer.

    Applie the dense layer and dropout to the output tensor, then adds the input tensor to the transformed output (residual connection).
    This is applied after the multi-head attention and the feed forward layer.

    Args:
      input (torch.Tensor): [bs, seq_len, hidden_size], the input to the sub-layer (before layer norm !).
      output (torch.Tensor): [bs, seq_len, hidden_size], the output from the sub-layer (before dropout and dense layer).
      dense_layer (nn.Linear): the linear transformation layer to be applied to the output.
      dropout (nn.Dropout): the dropout layer to be applied to the transformed output.

    Returns:
      torch.Tensor: [bs, seq_len, hidden_size] the result.
    """
    projected_output = dense_layer(output)
    projected_output = dropout(projected_output)
    return input + projected_output

  #NOTE: We implemented this
  def forward(self, hidden_states, attention_mask):
    """
    The forward pass of the GPT-2 layer.
            |
            |-----------
            |           |
         LayerNorm      |
            |           |
         Attention      |
            |           |
            |-----------
            |
            |-----------
            |           |
          LayerNorm     |
            |           |
          Linear        |
            |           |
        Activation f.   |  
            |           |
          Linear        |
            |           |
            |-----------|
            |
    Args:
      hidden_states (torch.Tensor): [bs, seq_len, hidden_size], the input hidden states to the GPT-2 layer.
      attention_mask (torch.Tensor): [bs, 1, 1, seq_len] mask which distinguishes between padded and non-padded tokens.
      
    Returns:
      torch.Tensor: [bs, seq_len, hidden_size], the output hidden states from the GPT-2 layer.
    """
    # 1. layer norm
    normed_hidden_states = self.attention_layer_norm(hidden_states)
    # 2. multi-head attention
    attn_output = self.self_attention(normed_hidden_states, attention_mask)
    # 3. skip connection and dropout
    attn_output = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)
    # 4. layer norm
    normed_attn_output = self.out_layer_norm(attn_output)
    # 5. final feed forward layer
    interm_output = self.interm_dense(normed_attn_output) # first layer
    interm_output = self.interm_af(interm_output) # activation function
    # 6. skip connection and dropout
    output = self.add(attn_output, interm_output, self.out_dense, self.out_dropout)
    return output

