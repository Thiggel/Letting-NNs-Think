import torch
from torch import nn
from typing import Any
from experiment.S6 import S6


class MambaTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, state_dimension: int = 128):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_size = None
        self.seq_len = None

        self.attention = nn.MultiheadAttention(d_model, nhead)

        self.state_dimension = state_dimension
        self.state = None
        self.s6 = S6(d_model, self.state_dimension)

        self.norm1 = nn.LayerNorm(self.state_dimension)
        self.norm2 = nn.LayerNorm(d_model)

    def get_device(self):
        return next(self.parameters()).device

    def squeeze_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_size, self.seq_len, self.d_model = x.shape

        return x.reshape(-1, self.d_model)

    def unsqueeze_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.batch_size, -1, self.d_model)

    def reset_state(self, x: torch.Tensor):
        self.state = torch.zeros(
            self.batch_size * self.seq_len, self.d_model, self.state_dimension
        ).to(self.get_device())

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, Any]:
        x = self.unsqueeze_seq_len(x)

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze().repeat(self.nhead, 1, 1)

        if attention_mask.shape[-1] != x.shape[-1]:
            attention_mask = None

        x = x.transpose(0, 1)
        attention_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        attention_output = attention_output.transpose(0, 1)
        x = x.transpose(0, 1)

        attention_output = self.squeeze_seq_len(attention_output)
        x = self.squeeze_seq_len(x)

        self.state, output = self.s6(self.state, self.norm2(attention_output))

        return output.squeeze(1), None
