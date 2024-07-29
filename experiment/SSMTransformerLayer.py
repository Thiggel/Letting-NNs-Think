from typing import Any
import torch
from torch import nn

from experiment.S6 import S6


class SSMTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.attention = nn.MultiheadAttention(d_model, nhead)

        self.S6 = S6(
            in_channels=d_model,
            hidden_dim=d_model,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, Any]:
        mask = attention_mask.squeeze().repeat(self.nhead, 1, 1)

        x = x.transpose(0, 1)
        attention_output, _ = self.attention(
            x, x, x, attn_mask=mask, is_causal=True, need_weights=False
        )
        x = x.transpose(0, 1)

        output = self.S6(hidden_states=x, inputs=attention_output)

        return (output, None)
