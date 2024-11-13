import math
import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        freq_embed_dim=256,
        max_period=10000,
    ):
        super().__init__()

        # Step 1: Create frequency buffer
        # This creates a geometric sequence from high to low frequencies
        # For freq_embed_dim=256, we get 128 frequencies going from ~1.0 to ~0.0001
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=freq_embed_dim // 2)
            / (freq_embed_dim // 2)
        )
        # Store as non-trainable buffer
        self.register_buffer("freqs", freqs)  # shape: (128,)

        # Step 4: Create MLP to process embeddings
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def get_freq_embedding(self, timestep: int) -> torch.Tensor:
        # Step 2: Create sinusoidal signals at each frequency
        # Multiply timestep by each frequency
        args = timestep * self.freqs  # shape: (128,)

        # Step 3: Get sine and cosine components
        # This creates pairs of values encoding each frequency
        embedding = torch.cat(
            [torch.cos(args), torch.sin(args)]  # shape: (128,)  # shape: (128,)
        )  # shape: (256,)

        return embedding

    def forward(self, hidden_states: torch.Tensor, timestep: int) -> torch.Tensor:
        # Get frequency embeddings
        t_freq = self.get_freq_embedding(timestep)  # shape: (256,)

        # Project to hidden dimension
        t_emb = self.mlp(t_freq)  # shape: (hidden_dim,)

        # Add to all positions
        return hidden_states + t_emb
