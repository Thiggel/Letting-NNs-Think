import torch
from torch import nn
from typing import Protocol


class MoDModelProtocol(Protocol):
    model: nn.Module

    def log(
        self,
        name: str,
        value: torch.Tensor,
        sync_dist: bool = False,
        batch_size: int = 1,
    ) -> None: ...


class MoDModel(MoDModelProtocol):
    def get_mod_loss(self) -> torch.Tensor:
        """Get the MoD loss"""
        if not hasattr(self.model, "mod_loss"):
            return torch.tensor(0)

        self.log("mod_loss", self.model.mod_loss)

        return self.model.mod_loss
