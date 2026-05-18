

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

        print(f'Initializing LoReFT intervention with rank {rank}')

        self.rank = rank

        # Low-rank rotation matrix: weight shape [hidden_size, rank], kept orthonormal via parametrization
        self.R_reft = nn.Linear(hidden_size, rank, bias=False)
        self.R_reft = torch.nn.utils.parametrizations.orthogonal(self.R_reft)

        # Learned linear transform in the subspace (no orthogonal constraint)
        self.W_reft = nn.Linear(hidden_size, rank)

    def forward(self, hidden_states, reft_mask):
        """
        hidden_states: [batch, seq_len, hidden_size]
        reft_mask:     [batch, seq_len] bool
        """

        selected = hidden_states[reft_mask]

        rotated = self.R_reft(selected)                                    # h @ R
        transformed = self.W_reft(selected)                                # W*h + b
        delta = (transformed - rotated) @ self.R_reft.weight            # R^T(Wh + b - Rh)

        # Clone necessary to not mess up gradients
        out = hidden_states.clone()
        out[reft_mask] = selected + delta

        return out



class DiReFTIntervention(nn.Module):
    def __init__(self, hidden_size, rank=4):
        super().__init__()

        self.rank = rank
        print(f"Initializing DiReFT intervention with rank {rank}")

        # Low-rank rotation matrix, kept orthonormal via parametrization
        self.R_reft = nn.Linear(hidden_size, rank, bias=False)
        self.R_reft = torch.nn.utils.parametrizations.orthogonal(self.R_reft)

        # Learned source (no orthogonal constraint)
        self.W_reft = nn.Linear(hidden_size, rank)

    def forward(self, hidden_states, reft_mask):
        """
        hidden_states: [batch, seq_len, hidden_size]
        reft_mask:     [batch, seq_len] bool
        """

        selected = hidden_states[reft_mask]                         # [N, hidden_size]

        transformed = self.W_reft(selected)                         # Wh + b, shape [N, rank]
        delta = transformed @ self.R_reft.weight                    # R^T(Wh + b), shape [N, hidden_size]

        # Clone necessary to not mess up gradients
        out = hidden_states.clone()
        out[reft_mask] = selected + delta

        return out