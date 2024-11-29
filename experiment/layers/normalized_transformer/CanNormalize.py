import torch


class CanNormalize:
    def normalize(self, x: torch.Tensor, dim=-1) -> torch.Tensor:
        return x / x.norm(p=2, dim=dim, keepdim=True)
