import torch
from torch import nn
from torch.nn import functional as F

from experiment.AdaptiveTransformer.AdaptiveLinear import AdaptiveLinear


class AdaptiveFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(AdaptiveFeedForward, self).__init__()
        self.W1 = AdaptiveLinear(d_model, d_ff)
        self.W2 = AdaptiveLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.relu(self.W1(x)))
