import torch
from torch import nn
from torchdeq import get_deq, reset_deq
from typing import Any
from transformers.cache_utils import DynamicCache


class RecurrentTransformerLayer(nn.Module):
    def __init__(
        self, layer: nn.Module, use_fixed_num_steps: bool = False, num_steps: int = 3
    ):
        super().__init__()
        self.layer = layer
        self.recurrence = get_deq(f_solver="fixed_point_iter")

        self.use_fixed_num_steps = use_fixed_num_steps
        self.num_steps = num_steps

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, Any]:
        if hasattr(self.layer, "squeeze_seq_len"):
            x = self.layer.squeeze_seq_len(x)

        past_key_value = kwargs.get("past_key_value", None)
        kwargs["past_key_value"] = None
        kwargs["use_cache"] = False

        if self.use_fixed_num_steps:
            for _ in range(self.num_steps):
                if (
                    attention_mask is not None
                    and attention_mask.shape[-1] != attention_mask.shape[-2]
                ):
                    attention_mask = None

                self.outputs = self.layer(
                    x, attention_mask=attention_mask, *args, **kwargs
                )
                x = self.outputs[0]

            output = x

        else:
            reset_deq(self.recurrence)

            def f(prev_hidden_states: torch.Tensor) -> torch.Tensor:
                self.outputs = self.layer(
                    prev_hidden_states,
                    attention_mask=attention_mask,
                    *args,
                    **kwargs,
                )

                hidden_states = self.outputs[0]

                hidden_states[torch.isnan(hidden_states)] = 0

                return hidden_states

            with torch.no_grad():
                fixed_points, _ = self.recurrence(f, x, tol=1e-2)

            output = fixed_points[-1]

        if hasattr(self.layer, "unsqueeze_seq_len"):
            output = self.layer.unsqueeze_seq_len(output)

        return output, past_key_value
