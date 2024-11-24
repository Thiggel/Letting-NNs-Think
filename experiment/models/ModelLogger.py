import torch
from lightning.pytorch import Trainer
from typing import Protocol, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel

from .MetricsLogger import MetricsLogger


class ModelLoggerProtocol(Protocol):
    metrics_logger: MetricsLogger
    trainer: Trainer
    tokenizer: Optional[PreTrainedTokenizer]
    model: PreTrainedModel


class ModelLogger(ModelLoggerProtocol):
    def on_before_optimizer_step(self, _):
        """Log gradient norms before optimization step"""
        self.metrics_logger.log_gradient_norms()

    def _dump_first_batch(self, batch: dict[str, torch.Tensor]) -> None:
        if self.trainer.global_rank > 0:
            return

        MAX_DUMPS = 5

        if not hasattr(self, "num_dumped_first_batch"):
            self.num_dumped_first_batch = 0

        if self.num_dumped_first_batch < MAX_DUMPS and self.tokenizer is not None:
            input_ids = batch["input_ids"]
            for i in range(min(len(input_ids), 3)):
                ids = input_ids[i]

                decoded = self.tokenizer.decode(ids)
                print()
                print(decoded)
                print()

            is_tied = (
                self.model.base_model.model.model.embed_tokens.weight.data_ptr()
                == self.model.base_model.model.lm_head.weight.data_ptr()
            )

            print(f"Embedding and LM head weights are tied: {is_tied}")
            print()

            self.num_dumped_first_batch += 1
