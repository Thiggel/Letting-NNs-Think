from pathlib import Path
import json
from experiment.models import DefaultLightningModule
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
from transformers import PreTrainedTokenizer
import os

from .CustomInference import CustomInference


class ModelEvaluator:
    """Handles model evaluation using lm-eval-harness with multi-GPU support"""

    def __init__(
        self,
        model: DefaultLightningModule,
        tokenizer: PreTrainedTokenizer,
        is_uninterrupted: bool = False,
        eval_batch_size: int = 128,
    ):
        self.eval_batch_size = eval_batch_size

        if is_uninterrupted:
            self.model = CustomInference(model, tokenizer)
        else:
            self.model = model

        self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self, metrics: list[str], seed: int, experiment_name: str
    ) -> dict[str, float]:
        wrapped_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=self.eval_batch_size,
            max_length=512,
            backend="causal",
            device=self.device,
            add_bos_token=True,
        )

        print("wiuhdwihdiuwhd")

        output = evaluator.simple_evaluate(
            model=wrapped_model,
            tasks=metrics or ["commonsense_qa", "gsm8k", "piqa"],
            num_fewshot=0,
            batch_size=self.eval_batch_size,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            device=self.device,
            log_samples=True,
        )

        print(output)

        self._save_results(output["results"], experiment_name)
        self._save_samples(output["samples"], seed, experiment_name)

        return output["results"]

    def _save_results(self, results: dict, experiment_name: str):
        output_dir = Path(os.getenv("BASE_CACHE_DIR") or "") / "evaluation_results"
        output_dir.mkdir(exist_ok=True)

        results_path = output_dir / f"{experiment_name}.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

    def _save_samples(self, samples: dict, seed, experiment_name):
        try:
            sample_dir = Path(os.environ["BASE_CACHE_DIR"] or "") / "samples"
            sample_dir.mkdir(exist_ok=True)

            sample_path = sample_dir / f"{experiment_name}_{seed}.json"
            sample_path.write_text(json.dumps(samples))
        except Exception as e:
            print(f"Failed to save samples: {e}")
