import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from experiment.AdaptiveTransformer.AdaptiveLinear import AdaptiveLinear


class AdaptiveMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(AdaptiveMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = AdaptiveLinear(d_model, d_model)
        self.W_k = AdaptiveLinear(d_model, d_model)
        self.W_v = AdaptiveLinear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.size()

        Q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        return self.W_o(output)
