import torch
from torch import nn
from typing import Optional


class RecurrentTransformerLayer(nn.Module):
    def __init__(self, layer: nn.Module, hidden_size: int, max_iter: int = 25):
        super().__init__()
        self.layer = layer
        self.exit_classifier = nn.Linear(hidden_size, 1)
        self.max_iter = max_iter

    def forward(
        self,
        initial_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Optional[tuple]:
        prev_hidden_states = initial_hidden_states

        output_hidden_states = torch.zeros_like(initial_hidden_states)

        for _ in range(self.max_iter):
            hidden_states = self.layer(
                prev_hidden_states, attention_mask=attention_mask, *args, **kwargs
            )[0]

            should_exit = torch.sigmoid(self.exit_classifier(hidden_states))

            output_hidden_states = torch.where(
                should_exit > 0.5, hidden_states, output_hidden_states
            )

            prev_hidden_states = hidden_states

        return output_hidden_states, None
