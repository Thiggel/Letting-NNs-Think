import torch
from torch import nn


class MoDRouter(nn.Module):
    def __init__(self, hidden_dim: int, router_hidden_dim: int):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, seq_len, hidden_dim]
        # Output shape: [batch_size, seq_len, 1]
        return self.router(hidden_states)
