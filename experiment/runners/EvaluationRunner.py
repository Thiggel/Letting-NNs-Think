from typing import Any, Dict
import wandb
import torch
from pydantic import BaseModel

from experiment.experiment import Runner
from experiment.experiment import ExperimentConfig
from experiment.configs import ModelConfig, DataConfig, TrainingConfig, EvaluationConfig
from experiment.model_evaluator import ModelEvaluator, SyntheticDatasetEvaluator

from .HasTokenizer import HasTokenizer
from .HasModel import HasModel


class EvaluationRunner(Runner, HasTokenizer, HasModel):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, seed: int) -> Dict[str, float]:
        model = self._load_model(seed, mode="test")
        model.to(self.device)

        string = self.tokenizer.encode("1 + 0 + 1 =", return_tensors="pt").to(
            self.device
        )
        generated = model.generate(
            input_ids=string,
            max_length=100,
            max_new_tokens=100,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        print("Sample generation: ", self.tokenizer.decode(generated[0]))

        # Determine if we're evaluating on synthetic datasets
        synthetic_tasks = [
            "arithmetic",
            "pattern",
        ]
        metrics = self.evaluation_config.evaluation_metrics or ["gsm8k"]

        results = {}

        # Handle synthetic dataset evaluation
        synthetic_metrics = [m for m in metrics if m in synthetic_tasks]
        if synthetic_metrics:
            evaluator = SyntheticDatasetEvaluator(
                model,
                self.tokenizer,
                self.evaluation_config.eval_batch_size,
                self.data_config,
                self.model_config,
                self.training_config,
                seed,
            )
            for task in synthetic_metrics:
                task_results = evaluator.evaluate(task)
                results.update({f"{task}_{k}": v for k, v in task_results.items()})

        # Handle standard dataset evaluation
        standard_metrics = [m for m in metrics if m not in synthetic_tasks]
        if standard_metrics:
            evaluator = ModelEvaluator(
                model,
                self.tokenizer,
                self.evaluation_config.evaluate_as_uninterrupted,
                self.evaluation_config.eval_batch_size,
                self.evaluation_config.num_fewshot,
            )
            standard_results = evaluator.evaluate(
                standard_metrics,
                seed,
                self.experiment_config.experiment_name,
            )
            results.update(self._format_standard_results(standard_results))

        if self.experiment_config.enable_logging:
            self._log_results(results, seed)

        return results

    def _log_results(self, results: Dict[str, Any], seed: int):
        wandb.init(
            project=self.experiment_config.project_name,
            name=f"{self.experiment_config.experiment_name}_{seed}",
            group=self.experiment_config.experiment_name,
        )
        wandb.log(results)
        wandb.finish()

    def _format_standard_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        return {
            f"{key}_accuracy": (
                value["acc,none"]
                if "acc,none" in value
                else value["exact_match,flexible-extract"]
            )
            for key, value in results.items()
        }
