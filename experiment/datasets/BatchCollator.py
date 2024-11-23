import torch
from typing import Optional
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
        max_unpadded_len = max(sum(item["attention_mask"]) for item in batch)

        input_ids = self._pad_and_truncate(
            [torch.tensor(item["input_ids"]) for item in batch],
            self.tokenizer.pad_token_id or 0,
            truncate_to=max_unpadded_len,
        )
        attention_mask = self._pad_and_truncate(
            [torch.tensor(item["attention_mask"]) for item in batch],
            0,
            truncate_to=max_unpadded_len,
        )
        labels = self._pad_and_truncate(
            [torch.tensor(item["labels"]) for item in batch],
            -100,
            truncate_to=max_unpadded_len,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _pad_and_truncate(
        self,
        sequences: list[torch.Tensor],
        padding_value: int,
        truncate_to: Optional[int] = None,
    ) -> torch.Tensor:
        padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        max_length = truncate_to or self.max_length
        truncated = padded[:, :max_length]

        return truncated
