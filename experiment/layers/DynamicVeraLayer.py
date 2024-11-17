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

        self.A = torch.empty(hidden_dim, vera_r).to(device).bfloat16()
        nn.init.kaiming_uniform_(self.A, mode="fan_in", nonlinearity="relu")

        self.B = torch.empty(vera_r, hidden_dim).to(device).bfloat16()
        nn.init.kaiming_uniform_(self.B, mode="fan_in", nonlinearity="relu")

        self.middle_scaling_vector_hypernet = nn.Linear(hidden_dim, vera_r).to(device)
        self.end_scaling_vector_hypernet = nn.Linear(hidden_dim, hidden_dim).to(device)

        self.layer = layer

    def forward(self, hidden_state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        middle_scaling_vector = self.middle_scaling_vector_hypernet(hidden_state)
        end_scaling_vector = self.end_scaling_vector_hypernet(hidden_state)

        vera_output = (hidden_state @ self.A) * middle_scaling_vector
        vera_output = (vera_output @ self.B) * end_scaling_vector

        return self.layer(hidden_state, *args, **kwargs) + vera_output
