from typing import Any, Dict
import numbers
import wandb
import torch
from torch import nn
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from optimum.quanto import QuantizedModelForCausalLM, qint4

from experiment.experiment import Runner
from experiment.experiment import ExperimentConfig
from experiment.configs import ModelConfig, DataConfig, TrainingConfig, EvaluationConfig
from experiment.model_evaluator import ModelEvaluator, ThresholdOptimizer
from experiment.models.mixture_of_depths import ModWrapper
from experiment.models.early_exit import EarlyExitWrapper

from .HasModel import HasModel
from .HasTokenizer import HasTokenizer


class EvaluationRunner(Runner):
    """Handles model evaluation, with an optional gating-threshold optimization phase."""

    def __init__(self, configs: dict[str, BaseModel]):
        super().__init__(configs)
        self.experiment_config: ExperimentConfig = self.configs[
            ExperimentConfig.__name__
        ]
        self.model_config: ModelConfig = self.configs[ModelConfig.__name__]
        self.data_config: DataConfig = self.configs[DataConfig.__name__]
        self.training_config: TrainingConfig = self.configs[TrainingConfig.__name__]
        self.evaluation_config: EvaluationConfig = self.configs[
            EvaluationConfig.__name__
        ]

    def run(self, seed: int, state_dict: torch.Tensor = None) -> Dict[str, float]:
        model = self._load_model(seed, mode="test").to(self.device)
        model.eval()

        # Phase 1: find optimal per-layer thresholds on a small subset
        subset_metric = [
            self.evaluation_config.subset_metric
        ]  # e.g. ['commonsense_qa']
        small_limit = self.evaluation_config.subset_limit
        evaluator_small = ModelEvaluator(
            model,
            self.tokenizer,
            self.evaluation_config.eval_batch_size,
            self.evaluation_config.num_fewshot,
            limit=small_limit,
        )

        # define evaluate_fn for optimizer: returns (compute_saved, accuracy)
        def eval_fn(x: torch.Tensor) -> Tuple[float, float]:
            # apply thresholds
            model.model_config.skip_threshold = x.tolist()
            # run subset evaluation
            results = evaluator_small.evaluate(
                metrics=subset_metric,
                seed=seed,
                experiment_name=f"thresh_opt_{seed}",
                generation_mode=self.model_config.generation_mode,
            )
            # get accuracy and compute saved
            acc = results[subset_metric[0]][self.evaluation_config.accuracy_key]
            pct_saved = sum(model.percent_tokens_skipped) / len(
                model.percent_tokens_skipped
            )
            return float(pct_saved), float(acc)

        num_layers = len(model.gating.layers)
        optimizer = ThresholdOptimizer(
            evaluate_fn=eval_fn,
            num_layers=num_layers,
            initial_samples=self.evaluation_config.initial_samples,
            device=self.device,
            dtype=torch.float32,
        )
        optimizer.run(iterations=self.evaluation_config.optim_iterations)
        # pick thresholds for desired savings s
        s = self.evaluation_config.target_savings
        optimal_t = optimizer.get_thresholds_for_s(s)
        model.gating.set_thresholds(optimal_t)

        # Phase 2: standard evaluation on all metrics with full limit
        evaluator_full = ModelEvaluator(
            model,
            self.tokenizer,
            self.evaluation_config.eval_batch_size,
            self.evaluation_config.num_fewshot,
            limit=self.evaluation_config.full_limit,
        )
        metrics = self.evaluation_config.evaluation_metrics
        for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            optimal_t = optimizer.get_thresholds_for_s(s)
            self.model_config.skip_threshold = optimal_t.tolist()
            # run subset evaluation
            results = evaluator_full.evaluate(
                metrics=metrics,
                seed=seed,
                experiment_name=f"{self.experiment_config.experiment_name}_thresh_opt_{seed}",
                generation_mode=self.model_config.generation_mode,
            )
            results = self._log_percent_tokens_skipped(model, results)
            results = self._log_percent_tokens_skipped_per_layer(model, results)

        return results
