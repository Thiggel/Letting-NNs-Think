import torch
from torch import nn
from torchdeq import get_deq
from typing import Optional


class RecurrentTransformerLayer(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.recurrence = get_deq()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Optional[tuple]:
        self.out = None

        def f(prev_last_hidden_state: torch.Tensor) -> torch.Tensor:
            sequence = torch.cat(
                [x[:, :-1, :], prev_last_hidden_state.unsqueeze(1)], dim=1
            )
            self.out = self.layer(sequence, *args, **kwargs)
            hidden_states = self.out[0]

            return hidden_states[:, -1, :]

        initial_state = x[:, -1, :]
        self.recurrence(f, initial_state, tol=1e-2)

        return self.out
