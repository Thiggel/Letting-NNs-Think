import torch

from .TokenManager import TokenManager


class ExitTokenManager(TokenManager):
    """Manages hidden states with interleaved exit tokens"""

    def prepare_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        exit_tokens = torch.zeros(
            batch_size, seq_len, hidden_size, device=hidden_states.device
        )
        return torch.stack((hidden_states, exit_tokens), dim=2).view(
            batch_size, seq_len * 2, hidden_size
        )

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = attention_mask.shape
        new_seq_len = seq_len * 2 if not inference_mode else seq_len + 1
        new_mask = torch.zeros(
            batch_size, new_seq_len, new_seq_len, device=attention_mask.device
        )

        if not inference_mode:
            for i in range(0, new_seq_len, 2):
                new_mask[:, i, : i + 1] = 1  # Normal token attends to previous tokens
                new_mask[:, i + 1, : i + 1] = 1  # Exit token attends to previous tokens
        else:
            new_mask[:, :-1, :-1] = attention_mask
            new_mask[:, -1, :-1] = 1  # Exit token attends to all previous tokens

        return new_mask.unsqueeze(1)

    def get_classifier_inputs(
        self, hidden_states: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor:
        return hidden_states[:, -1:] if inference_mode else hidden_states[:, 1::2]

    def finalize_hidden_states(
        self, hidden_states: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor:
        if not inference_mode:
            return hidden_states[:, ::2, :]  # Remove odd-indexed tokens (exit tokens)
        else:
            return hidden_states[:, :-1, :]  # Remove the last token (exit token)
