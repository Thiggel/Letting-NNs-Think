from typing import Protocol
import torch
from torch import nn

from transformers.models.gemma.modeling_gemma import GemmaConfig


class NormalizedDecoderLayerProtocol(Protocol):
    use_dynamic_rates: bool
    use_momentum: bool
    attn_alpha_init_value: float
    mlp_alpha_init_value: float
    attn_rate_predictor: nn.Linear
    mlp_rate_predictor: nn.Linear
    momentum_scale: nn.Parameter
    attn_momentum: torch.Tensor
    mlp_momentum: torch.Tensor
    momentum_decay: float
    attn_alpha: torch.Tensor
    mlp_alpha: torch.Tensor

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...


class NormalizedDecoderLayer:
    def setup(
        self: NormalizedDecoderLayerProtocol,
        config: GemmaConfig,
    ):
        if self.use_dynamic_rates:
            # Dynamic rate predictors
            self.attn_rate_predictor = nn.Linear(config.hidden_size, config.hidden_size)
            self.mlp_rate_predictor = nn.Linear(config.hidden_size, config.hidden_size)

            # Initialize close to static rates
            with torch.no_grad():
                self.attn_rate_predictor.bias.fill_(self.attn_alpha_init_value)
                self.mlp_rate_predictor.bias.fill_(self.mlp_alpha_init_value)

        if self.use_momentum:
            # Momentum parameters
            self.momentum_scale = nn.Parameter(torch.ones(config.hidden_size))
            self.register_buffer("attn_momentum", torch.zeros(config.hidden_size))
            self.register_buffer("mlp_momentum", torch.zeros(config.hidden_size))
            self.momentum_decay = 0.9  # Hyperparameter for momentum decay

    def get_eigen_rates(
        self: NormalizedDecoderLayerProtocol, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_dynamic_rates:
            return self.attn_alpha, self.mlp_alpha

        # Predict rates from current hidden state
        attn_rates = torch.sigmoid(
            self.attn_rate_predictor(hidden_states)
        )  # Sigmoid keeps rates positive
        mlp_rates = torch.sigmoid(self.mlp_rate_predictor(hidden_states))

        return attn_rates, mlp_rates

    def update_momentum(
        self: NormalizedDecoderLayerProtocol,
        current_delta: torch.Tensor,
        momentum_buffer: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_momentum:
            return current_delta

        # Update momentum as weighted sum of previous and current direction
        new_momentum = (
            self.momentum_decay * momentum_buffer
            + (1 - self.momentum_decay) * current_delta
        )

        # Store for next iteration
        momentum_buffer.copy_(new_momentum.detach())

        # Scale momentum by learnable parameter
        return self.momentum_scale * new_momentum
