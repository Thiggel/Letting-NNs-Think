import torch
from torch import nn

from experiment.AdaptiveTransformer.AdaptiveLinear import AdaptiveLinear


class AdaptiveSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        use_skip_connection: bool = False,
    ):
        super(AdaptiveSSM, self).__init__()

        self.A = AdaptiveLinear(d_model, d_model)

        self.B = AdaptiveLinear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.use_skip_connection = use_skip_connection

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        A_output = self.A(hidden_states)
        B_output = self.B(inputs)

        ssm_output = self.norm1(A_output) + B_output

        return self.norm2(ssm_output)
