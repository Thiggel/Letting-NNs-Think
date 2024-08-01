import torch
from torch import nn
from einops import einsum


class AdaptiveMatrix(nn.Module):
    def __init__(
        self,
        in_features: int,
        rows: int,
        cols: int,
    ):
        super(AdaptiveMatrix, self).__init__()

        self.rows = nn.Linear(in_features, rows)
        self.cols = nn.Linear(in_features, cols)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        get the rows of the matrix as a linear transormation of the input
        get the columns of the matrix as a linear transormation of the input
        then create a matrix of shape (n_rows x n_cols) by taking the outer product of
        the rows and columns
        """
        return einsum(
            self.rows(inputs),
            self.cols(inputs),
            "bsz seq_l rows, bsz seq_r cols -> bsz seq_l rows cols",
        )
