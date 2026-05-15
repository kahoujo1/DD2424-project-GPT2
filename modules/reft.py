

import torch
from torch import nn


def create_reft_mask(
    attention_mask: torch.Tensor,  # [batch, seq_len] or [seq_len]
    p: int,
    s: int,
    device: torch.device
) -> torch.BoolTensor:             # [batch, seq_len]
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    _, seq_len = attention_mask.shape
    n = attention_mask.sum(dim=-1)                          # [B] real token counts

    # Clamp p and s per sequence
    eff_p = torch.minimum(torch.full_like(n, p), n // 2)           # [B]
    eff_s = torch.minimum(torch.full_like(n, s), (n + 1) // 2)     # [B]

    # Position indices [1, seq_len] for broadcasting
    pos = torch.arange(seq_len, device=device).unsqueeze(0)         # [1, T]

    # For right-padded: real tokens are at positions [0, n)
    # Prefix: pos < eff_p
    # Suffix: pos >= (n - eff_s)
    prefix_mask = pos < eff_p.unsqueeze(1)                          # [B, T]
    suffix_mask = pos >= (n - eff_s).unsqueeze(1)                   # [B, T]

    # Only apply within real tokens
    reft_mask = (prefix_mask | suffix_mask) & attention_mask.bool()

    return reft_mask

class LoReFTIntervention(nn.Module):
    def __init__(self, hidden_size, rank=4):
        super().__init__()
        self.rank = rank

        # Low-rank rotation matrix (hidden_size -> rank)
        self.R = nn.Linear(hidden_size, rank, bias=False)
        # Learned linear transform in the subspace
        self.W = nn.Linear(rank, rank)
        
        # Optionally keep R orthonormal
        nn.init.orthogonal_(self.R.weight)

    def forward(self, hidden_states, reft_mask):
        """
        hidden_states: [batch, seq_len, hidden_size]
        reft_mask:     [seq_len] bool
        """
        selected = hidden_states[:, reft_mask, :]         # [B, T', H]

        projected = self.R(selected)                       # [B, T', rank]
        transformed = self.W(projected)                    # [B, T', rank]

        # Project back and compute delta
        delta = transformed @ self.R.weight                # [B, T', H]

        # Write back without mutating original
        out = hidden_states.clone()
        out[:, reft_mask, :] = selected + delta

        return out

class DiReFTIntervention(nn.Module):
    def __init__(self, hidden_size, rank=4):
        super().__init__()
        self.rank = rank

        # Two independent projection matrices (no weight tying)
        self.R1 = nn.Linear(hidden_size, rank, bias=False)
        self.R2 = nn.Linear(hidden_size, rank, bias=False)
        # Learned linear transform in the subspace
        self.W = nn.Linear(rank, rank)

        # No orthogonal init constraint in DiReFT
        nn.init.kaiming_uniform_(self.R1.weight)
        nn.init.kaiming_uniform_(self.R2.weight)

    def forward(self, hidden_states, reft_mask):
        """
        hidden_states: [batch, seq_len, hidden_size]
        reft_mask:     [seq_len] bool
        """
        selected = hidden_states[:, reft_mask, :]         # [B, T', H]

        projected = self.R1(selected)                      # [B, T', rank]
        transformed = self.W(projected)                    # [B, T', rank]

        # Project back with independent R2 (no weight tying)
        delta = transformed @ self.R2.weight               # [B, T', H]

        # Write back without mutating original
        out = hidden_states.clone()
        out[:, reft_mask, :] = selected + delta

        return out