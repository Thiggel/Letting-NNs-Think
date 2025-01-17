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
        num_steps: int,
    ):
        self.gemma_config = config
        if self.use_dynamic_rates:
            if self.use_lr_projection:
                self.init_gating()
            else:
                self.mlp_eigen_rates = nn.ModuleList(
                    [
                        nn.ParameterDict(
                            {
                                "rate": nn.Parameter(
                                    torch.randn(config.hidden_size) * 0.02
                                )
                            }
                        )
                        for _ in range(num_steps)
                    ]
                )

                self.attn_eigen_rates = nn.ModuleList(
                    [
                        nn.ParameterDict(
                            {"rate": torch.randn(config.hidden_size) * 0.02}
                        )
                        for _ in range(num_steps)
                    ]
                )

        if self.use_momentum:
            # Momentum parameters
            self.momentum_scale = nn.Parameter(torch.ones(config.hidden_size))
            self.register_buffer("attn_momentum", torch.zeros(config.hidden_size))
            self.register_buffer("mlp_momentum", torch.zeros(config.hidden_size))
            self.momentum_decay = 0.9  # Hyperparameter for momentum decay

    def init_gating(self):
        self.attn_rate_predictor = nn.Linear(
            self.gemma_config.hidden_size, self.gemma_config.hidden_size
        )
        self.mlp_rate_predictor = nn.Linear(
            self.gemma_config.hidden_size, self.gemma_config.hidden_size
        )

        # Initialize close to static rates
        with torch.no_grad():
            try:
                nn.init.normal_(self.attn_rate_predictor.weight, mean=0.0, std=0.02)
                self.attn_rate_predictor.bias.fill_(self.attn_alpha_init_value)
                nn.init.normal_(self.mlp_rate_predictor.weight, mean=0.0, std=0.02)
                self.mlp_rate_predictor.bias.fill_(self.mlp_alpha_init_value)
            except Exception:
                try:
                    nn.init.normal_(
                        self.attn_rate_predictor[0].weight, mean=0.0, std=0.02
                    )
                    self.attn_rate_predictor[0].bias.fill_(0.01)
                    nn.init.normal_(
                        self.attn_rate_predictor[-1].weight, mean=0.0, std=0.02
                    )
                    self.attn_rate_predictor[-1].bias.fill_(self.attn_alpha_init_value)

                    nn.init.normal_(
                        self.mlp_rate_predictor[0].weight, mean=0.0, std=0.02
                    )
                    self.mlp_rate_predictor[0].bias.fill_(0.01)
                    nn.init.normal_(
                        self.mlp_rate_predictor[-1].weight, mean=0.0, std=0.02
                    )
                    self.mlp_rate_predictor[-1].bias.fill_(self.mlp_alpha_init_value)
                except Exception:
                    pass

    def get_device(self):
        return next(self.parameters()).device

    def increment_timestep(self):
        new_mlp_rate = nn.ParameterDict(
            {
                "rate": nn.Parameter(
                    torch.randn(self.gemma_config.hidden_size, device=self.get_device())
                    * 0.02
                )
            }
        )
        self.mlp_eigen_rates.append(new_mlp_rate)

        new_attn_rate = nn.ParameterDict(
            {
                "rate": nn.Parameter(
                    torch.randn(self.gemma_config.hidden_size, device=self.get_device())
                    * 0.02
                )
            }
        )
        self.attn_eigen_rates.append(new_attn_rate)

    def get_mlp_eigen_rate(
        self: NormalizedDecoderLayerProtocol,
        hidden_states: torch.Tensor,
        mlp_output: torch.Tensor,
        timestep: int = 0,
    ) -> torch.Tensor:
        if not self.use_dynamic_rates:
            return self.mlp_alpha

        if self.use_lr_projection:
            mlp_rates = self.mlp_rate_predictor(mlp_output)

            return mlp_rates

        if timestep >= len(self.mlp_eigen_rates):
            self.increment_timestep()

        mlp_rates = self.mlp_eigen_rates[timestep]["rate"]

        return mlp_rates

    def get_attn_eigen_rate(
        self: NormalizedDecoderLayerProtocol,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor,
        timestep: int = 0,
    ) -> torch.Tensor:
        if not self.use_dynamic_rates:
            return self.attn_alpha

        if self.use_lr_projection:
            attn_rates = self.attn_rate_predictor(attention_output)

            return attn_rates

        if timestep >= len(self.attn_eigen_rates):
            self.increment_timestep()

        attn_rates = self.attn_eigen_rates[timestep]["rate"]

        return attn_rates

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
