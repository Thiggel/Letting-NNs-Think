import torch
from torch import nn
from torchdeq import get_deq, reset_deq
from typing import Any


class RecurrentTransformerLayer(nn.Module):
    def __init__(
        self, layer: nn.Module, use_fixed_num_steps: bool = False, num_steps: int = 3
    ):
        super().__init__()
        self.layer = layer
        self.recurrence = get_deq(f_solver="fixed_point_iter")

        self.use_fixed_num_steps = use_fixed_num_steps
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, Any]:
        if self.use_fixed_num_steps:
            for _ in range(self.num_steps):
                x = self.layer(x, *args, **kwargs)[0]

            return x, None

        reset_deq(self.recurrence)

        if hasattr(self.layer, "squeeze_seq_len"):
            x = self.layer.squeeze_seq_len(x)

        def f(prev_hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.layer(prev_hidden_states, *args, **kwargs)

            hidden_states[torch.isnan(hidden_states)] = 0

            return hidden_states

        fixed_points, _ = self.recurrence(f, x, tol=1e-2)

        if hasattr(self.layer, "unsqueeze_seq_len"):
            fixed_points = self.layer.unsqueeze_seq_len(fixed_points[-1])

        return fixed_points, None
