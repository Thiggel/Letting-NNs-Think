import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from einops import einsum


class S6(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        delta_low_rank_dim: Optional[int] = None,
    ):
        super(S6, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.delta_low_rank_dim: int = (
            delta_low_rank_dim
            if delta_low_rank_dim is not None
            else math.ceil(hidden_dim / 16)
        )

        self.A = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.A)

        self.delta_default = nn.Parameter(torch.empty(hidden_dim))
        nn.init.uniform_(self.delta_default, 0.001, 0.1)

        self.input_to_B_C_and_delta = nn.Linear(
            in_channels, self.delta_low_rank_dim + hidden_dim * 2
        )

        self.delta_proj = nn.Linear(self.delta_low_rank_dim, hidden_dim)

    def get_B_C_and_delta(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B_C_and_delta = self.input_to_B_and_delta(x)

        B, C, delta = torch.split(
            B_C_and_delta,
            [self.hidden_dim, self.hidden_dim, self.delta_low_rank_dim],
            dim=-1,
        )

        delta = F.softplus(self.delta_proj(delta) + self.delta_default)

        return B, C, delta

    def forward(
        self, hidden_states: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[-1] == self.hidden_dim, (
            f"Expected hidden_states to have shape (*, {self.hidden_dim}), "
            f"but got {hidden_states.shape}"
        )

        assert inputs.shape[-1] == self.in_channels, (
            f"Expected inputs to have shape (*, {self.in_channels}), "
            f"but got {inputs.shape}"
        )

        B, C, delta = self.get_B_C_and_delta(inputs)

        deltaA = delta.unsqueeze(-1) * self.A.unsqueeze(0)
        A_discrete = torch.exp(deltaA)
        A_output = A_discrete * hidden_states

        B_output = einsum(delta, B, inputs, "b d_in, b n, b d_in -> b d_in n")
        new_hidden_states = A_output + B_output
        C_output = einsum(new_hidden_states, C, "b d_in n, b n -> b d_in")

        return new_hidden_states, C_output
