import torch
from torch import nn
from typing import Optional, Tuple

from .RecurrenceStrategy import RecurrenceStrategy
from .FixedStepsStrategy import FixedStepsStrategy
from .TimestepEmbedder import TimestepEmbedder
from experiment.layers.normalized_transformer import NormalizedTimestepEmbedder
from .exit_classifier_strategy import ExitClassifierStrategy
from .FixedPointStrategy import FixedPointStrategy
from experiment.configs.ModelConfig import ModelConfig


class RecurrentTransformerLayer(nn.Module):
    """
    A transformer layer that can be made recurrent using different strategies.
    Supports fixed steps, adaptive exit classifier, or fixed point iteration.
    """

    def __init__(
        self,
        layer: nn.Module,
        config: ModelConfig,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.config = config
        self.layer = layer
        self.strategy = self._create_strategy(config, hidden_size, config.max_steps)
        self.intermediate_outputs = None

        if config.add_residual_connection and config.enable_normalization:
            self.alpha_init_value = 0.05
            self.alpha_init_scaling = 1.0 / (self.layer.config.hidden_size**0.5)
            self.alpha = torch.nn.Parameter(
                self.alpha_init_scaling
                * torch.ones(self.layer.config.hidden_size, dtype=torch.float32)
            )

    def _create_strategy(
        self, config: ModelConfig, hidden_size: int, max_steps: int
    ) -> RecurrenceStrategy:
        if config.use_time_embedding and config.enable_normalization:
            self.timestep_embedder = NormalizedTimestepEmbedder(hidden_size)
        elif config.use_time_embedding:
            self.timestep_embedder = TimestepEmbedder(hidden_size)
        else:
            self.timestep_embedder = None

        if config.num_steps is None:
            return FixedStepsStrategy(max_steps, hidden_size, config.use_time_embedding)

        if isinstance(config.num_steps, int):
            return FixedStepsStrategy(
                config.num_steps,
                config.use_time_embedding,
                self.timestep_embedder,
            )

        if config.num_steps == "classifier":
            return ExitClassifierStrategy(config, hidden_size)

        if config.num_steps == "fixed_point":
            return FixedPointStrategy()

        raise ValueError(f"Unknown recurrence mode: {config.num_steps}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Handle special layer requirements
        if hasattr(self.layer, "squeeze_seq_len"):
            hidden_states = self.layer.squeeze_seq_len(hidden_states)
        if hasattr(self.layer, "reset_state"):
            self.layer.reset_state(hidden_states)

        # Clear caching arguments
        kwargs["past_key_value"] = None
        kwargs["use_cache"] = False

        # Run the recurrence strategy
        output = self.strategy.forward(
            hidden_states, attention_mask, self.layer, position_ids, **kwargs
        )

        if self.config.add_residual_connection:
            output.hidden_states += hidden_states

        # Handle special layer requirements
        if hasattr(self.layer, "unsqueeze_seq_len"):
            output.hidden_states = self.layer.unsqueeze_seq_len(output.hidden_states)

        if self.config.add_residual_connection and self.config.enable_normalization:
            hidden_states = hidden_states + self.alpha * (
                output.hidden_states - hidden_states
            )

        elif self.config.add_residual_connection:
            hidden_states = hidden_states + output.hidden_states

        else:
            hidden_states = output.hidden_states

        self.intermediate_outputs = output.intermediate_outputs

        return (
            hidden_states,
            past_key_value,
            output.exit_probs,
        )
