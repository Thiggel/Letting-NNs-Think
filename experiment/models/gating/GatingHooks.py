from typing import Callable, Any
import torch
from functools import partial
from transformer_lens import HookedTransformer

from .TransformerGating import TransformerGating


class GatingHooks:
    """Manages TransformerLens hooks for gating"""

    @staticmethod
    def attention_hook(
        value: torch.Tensor, hook_self: Any, layer_idx: int, gating: TransformerGating
    ) -> torch.Tensor:
        if not gating.config.gate_attention:
            return value

        residual = hook_self.hook_normalize.input
        gate_value = gating.get_gate_value("attn", layer_idx, residual)
        return residual + gate_value * value

    @staticmethod
    def mlp_hook(
        value: torch.Tensor, hook_self: Any, layer_idx: int, gating: TransformerGating
    ) -> torch.Tensor:
        if not gating.config.gate_mlp:
            return value

        residual = hook_self.hook_normalize.input
        gate_value = gating.get_gate_value("mlp", layer_idx, residual)
        return residual + gate_value * value

    @staticmethod
    def add_hooks(
        model: HookedTransformer, gating: TransformerGating
    ) -> list[tuple[str, Callable]]:
        hooks = []

        for i in range(model.cfg.n_layers):
            if gating.config.gate_attention:
                hooks.append(
                    (
                        f"blocks.{i}.attn.hook_result",
                        partial(GatingHooks.attention_hook, layer_idx=i, gating=gating),
                    )
                )
            if gating.config.gate_mlp:
                hooks.append(
                    (
                        f"blocks.{i}.mlp.hook_result",
                        partial(GatingHooks.mlp_hook, layer_idx=i, gating=gating),
                    )
                )

        return hooks
