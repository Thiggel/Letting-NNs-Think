import torch
import random
from torch import nn
from torchdeq import get_deq, reset_deq
from typing import Any, Tuple, Optional


class RecurrentTransformerLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        use_fixed_num_steps: bool = False,
        use_random_num_steps: bool = False,
        num_steps: int = 10,
        use_time_embedding: bool = False,
        hidden_size: int = 768,  # Adjust this to match your model's hidden size
    ):
        super().__init__()
        self.layer = layer
        self.recurrence = get_deq(f_solver="fixed_point_iter")
        self.use_fixed_num_steps = use_fixed_num_steps
        self.use_random_num_steps = use_random_num_steps
        self.num_steps = num_steps
        self.use_time_embedding = use_time_embedding
        self.intermediate_outputs: list[torch.Tensor] = []

        # Exit token classifier
        self.exit_classifier = nn.Linear(hidden_size, 1)

    def _add_exit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        exit_tokens = torch.zeros(batch_size, seq_len, hidden_size, device=x.device)
        interleaved = torch.stack((x, exit_tokens), dim=2).view(
            batch_size, seq_len * 2, hidden_size
        )
        return interleaved

    def _create_exit_attention_mask(
        self, attention_mask: torch.Tensor, is_training: bool
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        new_seq_len = seq_len * 2 if is_training else seq_len + 1

        new_mask = torch.zeros(
            batch_size, new_seq_len, new_seq_len, device=attention_mask.device
        )

        if is_training:
            for i in range(0, new_seq_len, 2):
                new_mask[:, i, : i + 1] = (
                    1  # Normal token attends to all previous tokens
                )
                new_mask[:, i + 1, : i + 1] = (
                    1  # Exit token attends to all previous normal tokens
                )
        else:
            new_mask[:, :-1, :-1] = attention_mask
            new_mask[:, -1, :-1] = 1  # Exit token attends to all previous tokens

        return new_mask

    def _remove_exit_tokens(self, x: torch.Tensor, is_training: bool) -> torch.Tensor:
        if is_training:
            return x[:, ::2, :]  # Remove odd-indexed tokens (exit tokens)
        else:
            return x[:, :-1, :]  # Remove the last token (exit token)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        is_training = self.training

        if hasattr(self.layer, "squeeze_seq_len"):
            x = self.layer.squeeze_seq_len(x)
        if hasattr(self.layer, "reset_state"):
            self.layer.reset_state(x)

        past_key_value = kwargs.get("past_key_value", None)
        kwargs["past_key_value"] = None
        kwargs["use_cache"] = False

        x_with_exit = self._add_exit_tokens(x)
        new_attention_mask = self._create_exit_attention_mask(
            attention_mask, is_training
        )

        if is_training:
            if self.use_fixed_num_steps or self.use_random_num_steps:
                num_steps = (
                    self.num_steps
                    if self.use_fixed_num_steps
                    else random.randint(1, 10)
                )
                self.intermediate_outputs = []
                exit_probs = []

                for step in range(num_steps):
                    if self.use_time_embedding:
                        x_with_exit = x_with_exit + step + 1

                    self.outputs = self.layer(
                        x_with_exit, attention_mask=new_attention_mask, *args, **kwargs
                    )
                    x_with_exit = self.outputs[0]

                    # Compute exit probabilities
                    exit_logits = self.exit_classifier(x_with_exit[:, 1::2])
                    exit_prob = torch.sigmoid(exit_logits)
                    exit_probs.append(exit_prob)

                    if step < num_steps - 1:
                        self.intermediate_outputs.append(x_with_exit.clone())

                output = x_with_exit
                exit_probs = torch.cat(exit_probs, dim=1)
            else:
                reset_deq(self.recurrence)

                def f(prev_hidden_states: torch.Tensor) -> torch.Tensor:
                    self.outputs = self.layer(
                        prev_hidden_states,
                        attention_mask=new_attention_mask,
                        *args,
                        **kwargs,
                    )
                    hidden_states = self.outputs[0]
                    hidden_states[torch.isnan(hidden_states)] = 0
                    return hidden_states

                fixed_points, _ = self.recurrence(f, x_with_exit, tol=1e-2)
                output = fixed_points[-1]

                # Compute exit probabilities for the fixed point iteration case
                exit_logits = self.exit_classifier(output[:, 1::2])
                exit_probs = torch.sigmoid(exit_logits)
        else:
            # Inference mode (one token at a time)
            for step in range(self.num_steps):
                if self.use_time_embedding:
                    x_with_exit = x_with_exit + step + 1

                self.outputs = self.layer(
                    x_with_exit, attention_mask=new_attention_mask, *args, **kwargs
                )
                x_with_exit = self.outputs[0]

                # Compute exit probability for the last token
                exit_logits = self.exit_classifier(x_with_exit[:, -1:])
                exit_prob = torch.sigmoid(exit_logits)

                if exit_prob > 0.5:
                    break

            output = x_with_exit
            exit_probs = exit_prob

        output = self._remove_exit_tokens(output, is_training)

        if hasattr(self.layer, "unsqueeze_seq_len"):
            output = self.layer.unsqueeze_seq_len(output)

        return output, past_key_value, exit_probs
