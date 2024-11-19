from typing import Any, Optional
import torch
from transformers import PreTrainedTokenizer


class TokenizationProcessor:
    """Handles tokenization-related operations"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def tokenize_text(
        self, text: list[str], max_length: Optional[int] = None
    ) -> dict[str, list[Any]]:
        return self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=max_length is not None,
            max_length=max_length,
        )
