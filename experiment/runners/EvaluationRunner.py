from typing import Any
import wandb
import os
import torch
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

        evaluator = ModelEvaluator(
            model,
            self.tokenizer,
            self.evaluation_config.evaluate_as_uninterrupted,
            self.evaluation_config.eval_batch_size,
        )
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

            model = DefaultLightningModule(
                self.model_config,
                self.training_config,
                self.data_config,
                self.tokenizer,
            )
            model.setup("test")

            checkpoint = torch.load(checkpoint_path)

            state_dict = (
                checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            )
            print(state_dict.keys())
            print("-" * 80)
            print(model.state_dict().keys())
            exit()
            model.load_state_dict(state_dict)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            return model
        else:
            return DefaultLightningModule(
                self.model_config,
                self.training_config,
                self.data_config,
                self.tokenizer,
            )

    def _log_results(self, results: dict[str, Any], seed):
        wandb.init(
            project="variable-depth-lms3",
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
