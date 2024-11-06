from pathlib import Path
import json
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import os


class ModelEvaluator:
    """Handles model evaluation using lm-eval-harness"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(
        self, metrics: list[str], seed: int, experiment_name: str
    ) -> dict[str, float]:
        wrapped_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=16,
            max_length=512,
            backend="causal",
        )

        output = evaluator.simple_evaluate(
            model=wrapped_model,
            tasks=metrics or ["commonsense_qa", "gsm8k", "piqa"],
            num_fewshot=0,
            batch_size=16,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
            log_samples=True,
        )

        return output["results"]
