import torch
from torch import nn

from experiment.utils.hippo_init import hippo_init


class SSMTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        is_hippo_initialized: bool = False,
        use_norm: bool = False,
        use_adaptive_A: bool = True,
        use_adaptive_B: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.attention = nn.MultiheadAttention(d_model * nhead, nhead)
        self.q = nn.Linear(d_model, d_model * nhead)
        self.k = nn.Linear(d_model, d_model * nhead)
        self.v = nn.Linear(d_model, d_model * nhead)

        self.A_cols = nn.Linear(d_model, d_model, bias=False)
        self.A_rows = nn.Linear(d_model, d_model, bias=False)
        self.use_adaptive_A = use_adaptive_A

        self.B_cols = nn.Linear(d_model * nhead, d_model * nhead, bias=False)
        self.B_rows = nn.Linear(d_model * nhead, d_model, bias=False)
        self.use_adaptive_B = use_adaptive_B

        self.A = nn.Linear(d_model, d_model, bias=False)
        self.B = nn.Linear(d_model * nhead, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_norm else nn.Identity()

        if is_hippo_initialized:
            hippo_init(self.A, self.B)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor]:

        queries = self.q(x)
        keys = self.k(x)
        values = self.v(x)

        mask = attention_mask.squeeze().repeat(self.nhead, 1, 1)

        attention_output, _ = self.attention(
            queries, keys, values, attn_mask=mask, is_causal=True
        )

        if self.use_adaptive_B:
            B_cols = self.B_cols(attention_output)
            B_rows = self.B_rows(attention_output)
            B = B_cols.transpose(1, 2) @ B_rows

            B_output = self.norm1(torch.bmm(attention_output, B))
        else:
            B_output = self.norm1(self.B(attention_output))

        if self.use_adaptive_A:
            A_cols = self.A_cols(x)
            A_rows = self.A_rows(x)
            A = A_cols.transpose(1, 2) @ A_rows

            output = self.norm2(torch.bmm(x, A) + B_output)

        else:
            output = self.norm2(self.A(x) + B_output)

        return (output, None)
