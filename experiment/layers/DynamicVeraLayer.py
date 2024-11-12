import torch
from torch import nn


class DynamicVeraLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        hidden_dim: int,
        vera_r: int,
    ):
        super(DynamicVeraLayer, self).__init__()

        self.A = torch.empty(hidden_dim, vera_r)
        nn.init.kaiming_uniform_(self.A, mode='fan_in', nonlinearity='relu')

        self.B = torch.empty(hidden_dim, vera_r)
        nn.init.kaiming_uniform_(self.B, mode='fan_in', nonlinearity='relu')

        self.middle_scaling_vector_hypernet = nn.Linear(hidden_dim, vera_r)
        self.end_scaling_vector_hypernet = nn.Linear(hidden_dim, hidden_dim)

        self.layer = layer

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        middle_scaling_vector = self.middle_scaling_vector_hypernet(hidden_state)
        end_scaling_vector = self.end_scaling_vector_hypernet(hidden_state)

        vera_output = (self.A @ hidden_state) * middle_scaling_vector
        vera_output = (self.B @ vera_output) * end_scaling_vector

        return self.layer(hidden_state) + vera_output
