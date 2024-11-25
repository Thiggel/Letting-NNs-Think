from typing import Protocol, Optional
import torch
from torch import nn
import torch.nn.functional as F

from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import TrainingConfig

from .ModelAdapter import ModelAdapter


class RecurrentLanguageModelProtocol(Protocol):
    model_adapter: ModelAdapter
    model: nn.Module
    training_config: TrainingConfig

    def get_recurrent_layer(self) -> Optional[RecurrentTransformerLayer]: ...


class RecurrentLanguageModel(RecurrentLanguageModelProtocol):
    def get_recurrent_layer(self) -> Optional[RecurrentTransformerLayer]:
        """Get the recurrent layer if it exists"""
        if not hasattr(self.model_adapter, "recurrent_layer_idx"):
            return None
        return self.model.model.layers[self.model_adapter.recurrent_layer_idx]

    def get_loss_for_intermediate_supervision(self) -> torch.Tensor:
        layer = self.get_recurrent_layer()

        if (
            not self.training_config.use_random_intermediate_supervision
            or layer is None
            or not hasattr(layer, "intermediate_outputs")
            or layer.intermediate_outputs is None
            or len(layer.intermediate_outputs) == 0
        ):
            return torch.tensor(0)

        intermediate_outputs = torch.stack(layer.intermediate_outputs)

        loss = F.mse_loss(
            intermediate_outputs,
            torch.randn_like(intermediate_outputs),
        )

        return loss
