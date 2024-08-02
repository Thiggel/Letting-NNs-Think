import torch
from torch import nn
from typing import Optional

from experiment.AdaptiveTransformer.AdaptiveLinear import AdaptiveLinear
from experiment.AdaptiveTransformer.AdaptiveMultiHeadAttention import (
    AdaptiveMultiHeadAttention,
)


class AdaptiveTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(AdaptiveTransformerLayer, self).__init__()
        self.att = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.self_attention = AdaptiveMultiHeadAttention(d_model, num_heads)

        self.feed_forward = AdaptiveLinear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = num_heads

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        mask = attention_mask.repeat(1, self.nhead, 1, 1)
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return (x, None)
