import torch
from torch import nn
from einops import einsum

from experiment.AdaptiveTransformer.AdaptiveMatrix import AdaptiveMatrix


class AdaptiveLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super(AdaptiveLinear, self).__init__()

        self.matrix = AdaptiveMatrix(in_features, rows=in_features, cols=out_features)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        get the matrix as a transformation of the input,
        then multiply the input by the matrix
        """
        return einsum(
            inputs,
            self.matrix(inputs),
            "bsz seq_l r, bsz seq_l r c -> bsz seq_l c",
        )
