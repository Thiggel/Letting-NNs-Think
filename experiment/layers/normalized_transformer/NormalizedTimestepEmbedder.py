import math
import torch
from torch import nn
import torch.nn.functional as F


class NormalizedTimestepEmbedder(nn.Module):
    def __init__(
        self, hidden_dim: int, freq_embed_dim: int = 256, max_period: int = 10000
    ):
        super().__init__()

        # Create frequency buffer
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=freq_embed_dim // 2)
            / (freq_embed_dim // 2)
        )
        self.register_buffer("freqs", freqs)

        # MLP to process embeddings
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Learnable scaling factor for the update
        self.alpha = nn.Parameter(
            torch.ones(hidden_dim) * 0.1
        )  # Initialize conservatively

    def get_freq_embedding(self, timestep: int) -> torch.Tensor:
        args = timestep * self.freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)])
        return embedding

    def forward(self, hidden_states: torch.Tensor, timestep: int) -> torch.Tensor:
        # Get time embedding direction
        t_freq = self.get_freq_embedding(timestep)
        t_emb = self.mlp(t_freq)

        # Normalize the embedding direction
        t_emb = F.normalize(t_emb, dim=-1)

        # Update hidden state with learnable rate
        updated = hidden_states + self.alpha * (t_emb - hidden_states)

        # Project back to unit sphere
        return F.normalize(updated, dim=-1)
