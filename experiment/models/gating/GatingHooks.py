from typing import Callable, List, Any
import torch
from functools import partial
from transformer_lens.hook_points import HookPoint
from .TransformerGating import TransformerGating


class GatingHooks:
    """Manages TransformerLens hooks for gating"""

    @staticmethod
    def attention_hook(
        attn_output: torch.Tensor,
        hook: HookPoint,
        layer_idx: int,
        gating: TransformerGating,
    ) -> torch.Tensor:
        if not gating.config.gate_attention:
            return attn_output

        gate_value = gating.get_gate_value("attn", layer_idx, attn_output)
        return gate_value * attn_output

    @staticmethod
    def mlp_hook(
        mlp_output: torch.Tensor,
        hook: HookPoint,
        layer_idx: int,
        gating: TransformerGating,
    ) -> torch.Tensor:
        if not gating.config.gate_mlp:
            return mlp_output

        gate_value = gating.get_gate_value("mlp", layer_idx, mlp_output)
        return gate_value * mlp_output

    @staticmethod
    def add_hooks(model, gating: TransformerGating) -> None:
        """Add hooks directly using the model's hook_point system"""
        for i in range(model.cfg.n_layers):
            if gating.config.gate_attention:
                # Using hook_attn_out - this is after attn but before residual addition
                model.add_hook(
                    f"blocks.{i}.hook_attn_out",
                    partial(GatingHooks.attention_hook, layer_idx=i, gating=gating),
                )
            if gating.config.gate_mlp:
                # Using hook_mlp_out - this is after mlp but before residual addition
                model.add_hook(
                    f"blocks.{i}.hook_mlp_out",
                    partial(GatingHooks.mlp_hook, layer_idx=i, gating=gating),
                )
