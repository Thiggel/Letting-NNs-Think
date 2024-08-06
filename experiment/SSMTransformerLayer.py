import torch
from torch import nn
from typing import Any


class SSMTransformerLayer(nn.Module):
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

        self.A_cols = nn.Linear(d_model, d_model, bias=False)
        self.A_rows = nn.Linear(d_model, d_model, bias=False)

        self.B_cols = nn.Linear(d_model, d_model, bias=False)
        self.B_rows = nn.Linear(d_model, d_model, bias=False)

        self.A = nn.Linear(d_model, d_model, bias=False)
        self.B = nn.Linear(d_model * nhead, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def squeeze_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.d_model)

    def unsqueeze_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.batch_size, self.seq_len, self.d_model)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        x = self.unsqueeze_seq_len(x)

        mask = attention_mask.squeeze().repeat(self.nhead, 1, 1)

        x = x.transpose(0, 1)
        attention_output, _ = self.attention(x, x, x, attn_mask=mask)
        attention_output = attention_output.transpose(0, 1)
        x = x.transpose(0, 1)

        attention_output = self.squeeze_seq_len(attention_output)
        x = self.squeeze_seq_len(x)

        B_cols = self.B_cols(attention_output)
        B_rows = self.B_rows(attention_output)
        B = B_cols.unsqueeze(-1) @ B_rows.unsqueeze(-2)

        B_output = self.norm1(torch.bmm(attention_output.unsqueeze(1), B))

        A_cols = self.A_cols(x)
        A_rows = self.A_rows(x)
        A = A_cols.unsqueeze(-1) @ A_rows.unsqueeze(-2)

        output = self.norm2(torch.bmm(x.unsqueeze(1), A) + B_output)

        return output.squeeze(1)
