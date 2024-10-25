import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from typing import Any


class BatchCollator:
    """Handles batch collation operations"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = self._pad_and_truncate(
            [torch.tensor(item["input_ids"]) for item in batch],
            self.tokenizer.pad_token_id or 0,
        )
        attention_mask = self._pad_and_truncate(
            [torch.tensor(item["attention_mask"]) for item in batch], 0
        )
        labels = self._pad_and_truncate(
            [torch.tensor(item["labels"]) for item in batch], -100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _pad_and_truncate(
        self, sequences: list[torch.Tensor], padding_value: int
    ) -> torch.Tensor:
        padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        truncated = padded[:, : self.max_length]

        if truncated.shape[1] < self.max_length:
            pad_length = self.max_length - truncated.shape[1]
            truncated = F.pad(truncated, (0, pad_length), value=padding_value)

        return truncated
