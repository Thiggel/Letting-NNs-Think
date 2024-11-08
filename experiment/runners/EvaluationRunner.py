from typing import Any
from transformers import PreTrainedModel
import wandb
import os
from pydantic import BaseModel

from experiment.models import DefaultLightningModule
from experiment.experiment import Runner
from experiment.experiment import ExperimentConfig
from experiment.configs import ModelConfig, DataConfig, TrainingConfig, EvaluationConfig
from experiment.model_evaluator import ModelEvaluator

from .HasTokenizer import HasTokenizer


class EvaluationRunner(Runner, HasTokenizer):
    """Handles model evaluation"""

    def __init__(self, configs: dict[str, BaseModel]):
        super().__init__(configs)

        self.tokenizer = self._initialize_tokenizer()

        self.experiment_config: ExperimentConfig = self.configs[
            ExperimentConfig.__name__
        ]
        self.model_config: ModelConfig = self.configs[ModelConfig.__name__]
        self.data_config: DataConfig = self.configs[DataConfig.__name__]
        self.training_config: TrainingConfig = self.configs[TrainingConfig.__name__]
        self.evaluation_config: EvaluationConfig = self.configs[
            EvaluationConfig.__name__
        ]

    def run(self, seed: int) -> dict[str, float]:
        model = self._load_model(seed)
        evaluator = ModelEvaluator(model, self.tokenizer)
        results = evaluator.evaluate(
            self.evaluation_config.evaluation_metrics or ["gsm8k"],
            seed,
            self.experiment_config.experiment_name,
        )

        if self.experiment_config.enable_logging:
            self._log_results(results, seed)

        return self._format_results(results)

    def _load_model(self, seed: int) -> DefaultLightningModule:
        if self.evaluation_config.load_from_checkpoint:
            checkpoint_path = os.path.join(
                os.environ["BASE_CACHE_DIR"],
                f"{self.evaluation_config.load_from_checkpoint}_{seed}.pt",
            )
            print("Loading from checkpoint", checkpoint_path)
            return DefaultLightningModule.load_from_checkpoint(
                checkpoint_path,
                config=self.model_config,
                tokenizer=self.tokenizer,
                strict=False,
            )
        else:
            return DefaultLightningModule(self.model_config, self.tokenizer)

    def _log_results(self, results: dict[str, Any], seed):
        wandb.init(
            project="variable-depth-lms",
            name=f"{self.experiment_config.experiment_name}_{seed}",
            group=self.experiment_config.experiment_name,
        )
        wandb.log(results)
        wandb.finish()

    def _format_results(self, results: dict[str, Any]) -> dict[str, float]:
        return {
            f"{key}_accuracy": (
                value["acc,none"]
                if "acc,none" in value
                else value["exact_match,flexible-extract"]
            )
            for key, value in results.items()
        }
