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
    intermediate_outputs: torch.Tensor

    def get_recurrent_layer(self) -> Optional[RecurrentTransformerLayer]: ...

    def log(
        self,
        name: str,
        value: torch.Tensor,
        sync_dist: bool = False,
        batch_size: int = 1,
    ) -> None: ...


class RecurrentLanguageModel:
    def get_recurrent_layer(
        self: RecurrentLanguageModelProtocol,
    ) -> Optional[RecurrentTransformerLayer]:
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

    def get_loss_for_random_intermediate_supervision(
        self: RecurrentLanguageModelProtocol, layer: RecurrentTransformerLayer
    ) -> torch.Tensor:
        intermediate_outputs = torch.stack(layer.intermediate_outputs[:-1])
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
        random_targets = (
            eps * std[None, None, None, :]
            + self.random_target_mean[None, None, None, :]
        )

        # If using nGPT-style normalization, normalize the random targets
        if self.config.enable_normalization:
            random_targets = F.normalize(random_targets, dim=-1)

        loss = F.mse_loss(intermediate_outputs, random_targets)
        self.log("int_supervision_loss", loss, sync_dist=True, batch_size=batch_size)
        print(f"Intermediate supervision loss: {loss.item()}")
        print(f"Mean param max: {self.random_target_mean.max().item()}")
        print(f"Mean param min: {self.random_target_mean.min().item()}")
        print(f"Std param mean: {torch.exp(self.random_target_log_std).mean().item()}")

        reg_loss = 0.01 * (
            torch.sum(self.random_target_mean**2)
            + torch.sum(self.random_target_log_std**2)
        )

        return loss + reg_loss

    def get_loss_for_discounted_intermediate_supervision(
        self: RecurrentLanguageModelProtocol, layer: RecurrentTransformerLayer
    ) -> torch.Tensor:
        """
        Get loss for intermediate supervision using KL divergence between consecutive states
        and an optional entropy term to prevent collapse
        """
        intermediate_states = torch.stack(
            layer.intermediate_outputs[:-1]
        )  # [steps, batch, seq, hidden]
        next_states = torch.stack(
            layer.intermediate_outputs[1:]
        )  # [steps, batch, seq, hidden]

        # Calculate KL divergence between consecutive states
        # Since states are normalized (on hypersphere), we can use cosine similarity
        # to approximate the probability distribution
        cosine_sim = F.cosine_similarity(
            intermediate_states, next_states, dim=-1
        )  # [steps, batch, seq]
        # Scale to [0,1] range for probability interpretation

        # KL term: We want consecutive states to be similar but not identical
        # Using negative cosine similarity as a proxy for KL
        int_state_sim_loss = -cosine_sim.mean()

        # Log metrics if logging is available
        if hasattr(self, "log"):
            batch_size = intermediate_states.shape[1]
            self.log(
                "intermediate_state_similarity_loss",
                int_state_sim_loss,
                sync_dist=True,
                batch_size=batch_size,
            )

        return (
            self.training_config.intermediate_supervision_loss_weight
            * int_state_sim_loss
        )

    def get_loss_for_intermediate_supervision(
        self: RecurrentLanguageModelProtocol,
    ) -> torch.Tensor:
        layer = self.get_recurrent_layer()

        if (
            layer is None
            or not hasattr(layer, "intermediate_outputs")
            or layer.intermediate_outputs is None
            or len(layer.intermediate_outputs) == 0
        ):
            return torch.tensor(0)

        if self.training_config.use_discounted_intermediate_supervision:
            return self.get_loss_for_discounted_intermediate_supervision(layer)

        if self.training_config.use_random_intermediate_supervision:
            return self.get_loss_for_random_intermediate_supervision(layer)

        return torch.tensor(0)
