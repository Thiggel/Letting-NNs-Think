from typing import Protocol, Optional
import torch
from torch import nn
import torch.nn.functional as F

from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import ModelConfig, TrainingConfig

from .model_adapter import ModelAdapter


class RecurrentLanguageModelProtocol(Protocol):
    model_adapter: ModelAdapter
    model: nn.Module
    config: ModelConfig
    training_config: TrainingConfig

    def get_recurrent_layer(self) -> Optional[RecurrentTransformerLayer]: ...

    def log(
        self,
        name: str,
        value: torch.Tensor,
        sync_dist: bool = False,
        batch_size: int = 1,
    ) -> None: ...


class RecurrentLanguageModel(RecurrentLanguageModelProtocol):
    def get_recurrent_layer(self) -> Optional[RecurrentTransformerLayer]:
        """Get the recurrent layer if it exists"""
        if not hasattr(self.model_adapter, "recurrent_layer_idx"):
            return None
        layers = self.model_adapter.get_decoder_layers(self.model)
        return layers[self.model_adapter.recurrent_layer_idx]

    def setup_random_intermediate_supervision(self) -> None:
        if self.training_config.use_random_intermediate_supervision:
            self.random_target_mean = nn.Parameter(
                torch.zeros(self.model.config.hidden_size)
            )
            self.random_target_log_std = nn.Parameter(
                torch.ones(self.model.config.hidden_size)
            )

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
        batch_size, num_steps, seq_len, hidden_size = intermediate_outputs.shape

        # Create random noise
        eps = torch.randn(
            batch_size,
            num_steps,
            seq_len,
            hidden_size,
            device=intermediate_outputs.device,
            dtype=intermediate_outputs.dtype,
            requires_grad=True,
        )

        # Transform noise using learned parameters
        std = torch.exp(self.random_target_log_std)
        random_targets = eps * std + self.random_target_mean

        # If using nGPT-style normalization, normalize the random targets
        if self.config.enable_normalization:
            random_targets = F.normalize(random_targets, dim=-1)

        loss = F.mse_loss(intermediate_outputs, random_targets)
        self.log("int_supervision_loss", loss, sync_dist=True, batch_size=batch_size)
        print(f"Intermediate Supervision Loss: {loss.item()}")

        reg_loss = 0.01 * (
            torch.sum(self.random_target_mean**2)
            + torch.sum(self.random_target_log_std**2)
        )

        return loss + reg_loss
