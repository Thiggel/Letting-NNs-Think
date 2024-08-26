import torch
from torch import nn
from typing import Any
from experiment.S6 import S6


class MambaTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        batch_size: int,
        seq_len: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.attention = nn.MultiheadAttention(d_model, nhead)

        self.state_dimension = d_model
        self.state = torch.zeros((self.batch_size * self.seq_len))
        self.s6 = S6(d_model, self.state_dimension)

        self.norm = nn.LayerNorm(d_model)

    def squeeze_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_size is None or self.seq_len is None:
            self.batch_size, self.seq_len, self.d_model = x.shape

        return x.reshape(-1, self.d_model, self.state_dimension)

    def unsqueeze_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.batch_size, -1, self.d_model, self.state_dimension)

    def expand_state_dimension(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 1, self.state_dimension)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, Any]:

        x = self.unsqueeze_seq_len(x)

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze().repeat(self.nhead, 1, 1)

        x = x.transpose(0, 1)
        attention_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        attention_output = attention_output.transpose(0, 1)
        x = x.transpose(0, 1)

        attention_output = self.squeeze_seq_len(attention_output)
        x = self.squeeze_seq_len(x)

        new_hidden_states, output = self.norm(self.s6(x, attention_output))

        return output.squeeze(1), None
