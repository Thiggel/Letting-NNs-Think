import torch
from torch import nn


class DynamicVeraLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        hidden_dim: int,
        vera_r: int,
        device: torch.device,
    ):
        super(DynamicVeraLayer, self).__init__()

        # Initialize A and B matrices with careful scaling
        scaling = 1.0 / (hidden_dim**0.5)
        A = torch.empty(hidden_dim, vera_r).to(device=device)
        nn.init.normal_(A, std=scaling)

        B = torch.empty(vera_r, hidden_dim).to(device=device)
        nn.init.normal_(B, std=scaling)

        self.register_buffer("A", A)
        self.register_buffer("B", B)

        # Smaller hypernets with carefully chosen dimensions
        hidden_factor = 4  # Reduce from default size
        self.middle_scaling_vector_hypernet = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // hidden_factor),
            nn.LayerNorm(hidden_dim // hidden_factor),
            nn.GELU(),
            nn.Linear(hidden_dim // hidden_factor, vera_r),
            nn.LayerNorm(vera_r),
        ).to(device)

        self.end_scaling_vector_hypernet = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // hidden_factor),
            nn.LayerNorm(hidden_dim // hidden_factor),
            nn.GELU(),
            nn.Linear(hidden_dim // hidden_factor, hidden_dim),
            nn.LayerNorm(hidden_dim),
        ).to(device)

        # Learnable scaling factor for the VeRA contribution
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        self.layer = layer

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Generate and normalize scaling vectors
        middle_scaling_vector = self.middle_scaling_vector_hypernet(hidden_state)
        end_scaling_vector = self.end_scaling_vector_hypernet(hidden_state)

        # Apply VeRA with controlled scaling
        vera_output = self.dropout(hidden_state @ self.A)
        vera_output = vera_output * middle_scaling_vector
        vera_output = self.dropout(vera_output @ self.B)
        vera_output = vera_output * end_scaling_vector

        # Scale the VeRA contribution
        vera_output = vera_output * torch.sigmoid(self.output_scale)

        # Combine with layer output using residual connection
        layer_output = self.layer(hidden_state, *args, **kwargs)

        return layer_output + vera_output
