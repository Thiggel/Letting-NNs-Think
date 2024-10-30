from typing import Protocol
import torch


class AdvantageEstimator(Protocol):
    def estimate(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor: ...
