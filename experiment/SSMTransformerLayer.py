import torch
from torch import nn

from experiment.utils.hippo_init import hippo_init


class SSMTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        is_hippo_initialized: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.attention = nn.MultiheadAttention(d_model * nhead, nhead)
        self.q = nn.Linear(d_model, d_model * nhead)
        self.k = nn.Linear(d_model, d_model * nhead)
        self.v = nn.Linear(d_model, d_model * nhead)

        self.A = nn.Linear(d_model, d_model, bias=False)
        self.B = nn.Linear(d_model * nhead, d_model, bias=False)

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

        output = self.A(x) + self.B(attention_output)

        return (output, None)
