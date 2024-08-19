import torch
from torch import nn
from typing import Any


class SSMTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_size = None
        self.seq_len = None

        self.attention = nn.MultiheadAttention(d_model, nhead)

        self.rank = 2

        self.A_cols = nn.Linear(d_model, self.rank * d_model, bias=False)
        self.A_rows = nn.Linear(d_model, self.rank * d_model, bias=False)

        self.B_cols = nn.Linear(d_model, self.rank * d_model, bias=False)
        self.B_rows = nn.Linear(d_model, self.rank * d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def squeeze_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_size is None or self.seq_len is None:
            self.batch_size, self.seq_len, self.d_model = x.shape

        return x.reshape(-1, self.d_model)

    def unsqueeze_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.batch_size, -1, self.d_model)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, Any]:

        x = self.unsqueeze_seq_len(x)

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze().repeat(self.nhead, 1, 1)

            if attention_mask.shape[1] != attention_mask.shape[2]:
                mask = None

        x = x.transpose(0, 1)
        attention_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        attention_output = attention_output.transpose(0, 1)
        x = x.transpose(0, 1)

        attention_output = self.squeeze_seq_len(attention_output)
        x = self.squeeze_seq_len(x)

        B_cols = self.B_cols(attention_output).reshape(-1, self.d_model, self.rank)
        B_rows = self.B_rows(attention_output).reshape(-1, self.rank, self.d_model)
        B = B_cols @ B_rows

        B_output = self.norm1(torch.bmm(attention_output.unsqueeze(-2), B))

        A_cols = self.A_cols(x).reshape(-1, self.d_model, self.rank)
        A_rows = self.A_rows(x).reshape(-1, self.rank, self.d_model)
        A = A_cols @ A_rows

        output = self.norm2(torch.bmm(x.unsqueeze(1), A) + B_output)

        return output.squeeze(1), None
