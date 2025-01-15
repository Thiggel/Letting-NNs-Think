import math
import torch
from torch import nn
import torch.nn.functional as F


class NormalizedTimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, max_steps: int = 32):
        super().__init__()
        # Create a set of learnable basis vectors on the hypersphere
        num_basis = min(hidden_dim // 2, 32)
        self.basis = nn.Parameter(torch.randn(num_basis, hidden_dim))
        self.alpha = nn.Parameter(torch.ones(hidden_dim) * 0.05)
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim

    def forward(self, hidden_states: torch.Tensor, timestep: int) -> torch.Tensor:
        # Normalize basis vectors
        basis = F.normalize(self.basis, dim=-1)  # [num_basis, hidden_dim]

        # Create step-specific angles for all basis vectors at once
        # timestep shape: [batch_size]
        batch_size, seq_len = hidden_states.shape[0:2]
        step_fractions = (
            torch.tensor(
                [timestep / self.max_steps],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            .unsqueeze(-1)
            .repeat(batch_size, seq_len)
        )  # [batch_size, seq_len]

        # Create frequency matrix for all basis vectors
        freqs = 2.0 ** -torch.arange(self.basis.shape[0], device=hidden_states.device)
        # [num_basis]

        # Compute all angles at once
        # [batch_size, seq_len, num_basis]
        angles = step_fractions.unsqueeze(-1) * freqs.unsqueeze(0) * math.pi

        # Get coefficients for all bases at once
        # [batch_size, seq_len, num_basis]
        coeffs = torch.sin(angles)

        # Combine basis vectors according to coefficients
        # [batch_size, num_basis, 1] * [1, num_basis, hidden_dim]
        # -> [batch_size, seq_len, hidden_dim]
        direction = torch.sum(coeffs.unsqueeze(-1) * basis.unsqueeze(0), dim=2)

        # Normalize the direction vectors
        direction = F.normalize(direction, dim=-1)

        # Update hidden states using nGPT update equation
        # h ← Norm(h + α(hA − h))
        updated = hidden_states + self.alpha * (direction - hidden_states)
        return F.normalize(updated, dim=-1)
