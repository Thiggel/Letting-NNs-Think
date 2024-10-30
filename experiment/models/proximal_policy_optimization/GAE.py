import torch

from .AdvantageEstimator import AdvantageEstimator


class GAE(AdvantageEstimator):
    def __init__(self, discount_factor: float, gae_lambda: float):
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda

    def estimate(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.discount_factor * values[t + 1] - values[t]
            gae = delta + self.discount_factor * self.gae_lambda * gae
            advantages[t] = gae

        return advantages
