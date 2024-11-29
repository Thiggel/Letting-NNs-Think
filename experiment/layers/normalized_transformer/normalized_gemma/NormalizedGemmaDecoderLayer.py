from typing import Optional, Tuple
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.gemma.modeling_gemma import GemmaConfig
from transformers.utils import logging

from ..CanNormalize import CanNormalize
from ..NormalizedDecoderLayer import NormalizedDecoderLayer
from .NormalizedGemmaMLP import NormalizedGemmaMLP
from .NormalizedGemmaSdpaAttention import NormalizedGemmaSdpaAttention

logger = logging.get_logger(__name__)


class NormalizedGemmaDecoderLayer(nn.Module, CanNormalize, NormalizedDecoderLayer):
    def __init__(
        self,
        config: GemmaConfig,
        layer_idx: int,
        use_dynamic_rates: bool = False,
        use_momentum: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_dynamic_rates = use_dynamic_rates
        self.use_momentum = use_momentum

        # Core components
        self.self_attn = NormalizedGemmaSdpaAttention(
            config=config, layer_idx=layer_idx
        )
        self.mlp = NormalizedGemmaMLP(config)

        # Static eigen learning rates (used when not dynamic)
        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / (config.hidden_size**0.5)
        self.attn_alpha = torch.nn.Parameter(
            self.attn_alpha_init_scaling
            * torch.ones(config.hidden_size, dtype=torch.float32)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / (config.hidden_size**0.5)
        self.mlp_alpha = torch.nn.Parameter(
            self.mlp_alpha_init_scaling
            * torch.ones(config.hidden_size, dtype=torch.float32)
        )

        self.setup(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        # Get eigen learning rates (either static or dynamic)
        attn_rates, mlp_rates = self.get_eigen_rates(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = self.normalize(hidden_states)

        # Calculate attention update with optional momentum
        attn_delta = hidden_states - residual
        if self.use_momentum:
            attn_delta = self.update_momentum(attn_delta, self.attn_momentum)

        hidden_states = self.normalize(residual + attn_rates * attn_delta)

        # MLP
        residual = hidden_states
        hidden_states = self.normalize(self.mlp(hidden_states))

        # Calculate MLP update with optional momentum
        mlp_delta = hidden_states - residual
        if self.use_momentum:
            mlp_delta = self.update_momentum(mlp_delta, self.mlp_momentum)

        hidden_states = self.normalize(residual + mlp_rates * mlp_delta)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
