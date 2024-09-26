import torch
import random
from torch import nn
from torchdeq import get_deq, reset_deq
from typing import Any, Tuple, Optional

from experiment.utils import Args


class RecurrentTransformerLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        args: Args,
        max_steps: int = 20,
        hidden_size: int = 768,
        inference_mode=False,
    ):
        super().__init__()

        self.layer = layer

        self.use_fixed_num_steps = type(args.num_steps) == int
        self.use_random_num_steps = args.num_steps == "random"
        self.use_classifier = args.num_steps == "classifier"
        self.use_fixed_point = args.num_steps == "fixed_point"
        self.num_steps = args.num_steps if self.use_fixed_num_steps else max_steps
        self.use_time_embedding = args.use_time_embedding

        self.intermediate_outputs: list[torch.Tensor] = []

        self.recurrence = get_deq(f_solver="fixed_point_iter")

        self.exit_classifier = nn.Linear(hidden_size, 1)

        self.exit_probs = None

        self.inference_mode = inference_mode
        torch.autograd.set_detect_anomaly(True)

    def _add_exit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_classifier:
            return x

        batch_size, seq_len, hidden_size = x.shape
        exit_tokens = torch.zeros(batch_size, seq_len, hidden_size, device=x.device)
        interleaved = torch.stack((x, exit_tokens), dim=2).view(
            batch_size, seq_len * 2, hidden_size
        )
        return interleaved

    def _create_exit_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if not self.use_classifier:
            return attention_mask

        batch_size, _, seq_len, _ = attention_mask.shape
        new_seq_len = seq_len * 2 if not self.inference_mode else seq_len + 1

        new_mask = torch.zeros(
            batch_size, new_seq_len, new_seq_len, device=attention_mask.device
        )

        if not self.inference_mode:
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

        return new_mask.unsqueeze(1)

    def _remove_exit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_classifier:
            return x

        if not self.inference_mode:
            return x[:, ::2, :]  # Remove odd-indexed tokens (exit tokens)
        else:
            return x[:, :-1, :]  # Remove the last token (exit token)

    def deq_forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
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

        fixed_points, _ = self.recurrence(f, x, tol=1e-2)
        output = fixed_points[-1]

        return output

    def recurrent_forward_train(
        self,
        x_with_exit: torch.Tensor,
        new_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_steps = (
            self.num_steps if not self.use_random_num_steps else random.randint(1, 10)
        )
        self.intermediate_outputs = []
        exit_probs = []

        position_ids = torch.repeat_interleave(position_ids, repeats=2).unsqueeze(0)

        # Create an exit mask initialized to all False (i.e., no tokens have exited yet)
        batch_size, seq_len, hidden_size = x_with_exit.shape
        exit_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=x_with_exit.device
        )

        # Store the hidden states of tokens that have already exited (detached)
        frozen_hidden_states = torch.zeros_like(x_with_exit)

        for step in range(num_steps):
            if self.use_time_embedding:
                # Avoid in-place operation by reassigning instead of +=
                x_with_exit = x_with_exit + (step + 1)

            # Process the current step's forward pass
            self.outputs = self.layer(
                x_with_exit,
                attention_mask=new_attention_mask,
                position_ids=position_ids,
                *args,
                **kwargs,
            )
            x_with_exit = self.outputs[0]

            # Compute exit probabilities for classifier tokens (every second token)
            if self.use_classifier:
                exit_logits = self.exit_classifier(
                    x_with_exit[:, 1::2]
                )  # Classifier tokens
                exit_prob = torch.sigmoid(exit_logits)
                exit_probs.append(exit_prob)

                # Sample exit decisions from Bernoulli distribution based on the probabilities
                exit_decisions = torch.bernoulli(exit_prob).bool()

                # Create a new exit mask instead of modifying it in place
                new_exit_mask = exit_mask.clone()
                new_exit_mask[:, 1::2] = new_exit_mask[
                    :, 1::2
                ] | exit_decisions.squeeze(-1)
                new_exit_mask[:, ::2] = new_exit_mask[
                    :, 1::2
                ]  # Update corresponding original tokens

                # Assign back to the original mask (this is safe)
                exit_mask = new_exit_mask

            # Detach the hidden states of exited tokens to stop further gradient updates
            frozen_hidden_states = torch.where(
                exit_mask.unsqueeze(
                    -1
                ),  # Mask applies to both original and exit tokens
                x_with_exit.detach(),  # Detach the hidden states of exited tokens
                frozen_hidden_states,  # Continue using the previous frozen states
            )

            # Update x_with_exit only for the tokens that haven't exited
            x_with_exit = torch.where(
                exit_mask.unsqueeze(-1),
                frozen_hidden_states,  # Replace the exited tokens with their detached states
                x_with_exit,  # Continue updating tokens that haven't exited
            )

            if step < num_steps - 1:
                self.intermediate_outputs.append(x_with_exit.clone())

        output = x_with_exit
        exit_probs = torch.cat(exit_probs, dim=1)

        return output, exit_probs

    def recurrent_forward_inference(
        self,
        x_with_exit: torch.Tensor,
        new_attention_mask: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode (one token at a time)
        """
        exit_prob = torch.Tensor(0.0)

        batch_size, seq_len, hidden_size = x_with_exit.shape
        exit_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=x_with_exit.device
        )

        for step in range(self.num_steps):
            if self.use_time_embedding:
                x_with_exit = x_with_exit + step + 1

            # Forward pass
            self.outputs = self.layer(
                x_with_exit, attention_mask=new_attention_mask, *args, **kwargs
            )
            x_with_exit = self.outputs[0]

            # Compute exit probabilities for the last token (exit token)
            exit_logits = self.exit_classifier(x_with_exit[:, -1:])
            exit_prob = torch.sigmoid(exit_logits)

            # Sample exit decision from the classifier's probability distribution
            exit_decision = torch.bernoulli(exit_prob).bool()

            # If the token exits, mark it and its corresponding exit token
            if exit_decision:
                exit_mask[:, -1] = True  # Mark both token and exit token as exited
                break

            # Keep exited tokens and their exit tokens unchanged in the sequence
            x_with_exit = torch.where(
                exit_mask.unsqueeze(-1),
                x_with_exit.detach(),  # Keep exited tokens' states unchanged
                x_with_exit,  # Continue updating the rest
            )

        output = x_with_exit

        return output, exit_prob

    def recurrent_forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_with_exit = self._add_exit_tokens(x)
        new_attention_mask = self._create_exit_attention_mask(attention_mask)

        if not self.inference_mode:
            output, exit_probs = self.recurrent_forward_train(
                x_with_exit, new_attention_mask, *args, **kwargs
            )
        else:
            output, exit_probs = self.recurrent_forward_inference(
                x_with_exit, new_attention_mask, *args, **kwargs
            )

        output = self._remove_exit_tokens(output)

        return output, exit_probs

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        if hasattr(self.layer, "squeeze_seq_len"):
            x = self.layer.squeeze_seq_len(x)
        if hasattr(self.layer, "reset_state"):
            self.layer.reset_state(x)

        past_key_value = kwargs.get("past_key_value", None)
        kwargs["past_key_value"] = None
        kwargs["use_cache"] = False

        exit_probs = None

        if self.use_fixed_point:
            output = self.deq_forward(x, attention_mask, *args, **kwargs)
        else:
            output, exit_probs = self.recurrent_forward(
                x, attention_mask, *args, **kwargs
            )

        if hasattr(self.layer, "unsqueeze_seq_len"):
            output = self.layer.unsqueeze_seq_len(output)

        self.exit_probs = exit_probs

        return output, past_key_value, exit_probs
