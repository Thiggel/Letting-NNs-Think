from typing import Any, Dict, Tuple
import numbers
import wandb
import torch
from torch import nn
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from optimum.quanto import QuantizedModelForCausalLM, qint4
from tqdm import tqdm

from experiment.experiment import Runner
from experiment.experiment import ExperimentConfig
from experiment.configs import ModelConfig, DataConfig, TrainingConfig, EvaluationConfig
from experiment.model_evaluator import ModelEvaluator, ThresholdOptimizer
from experiment.models.mixture_of_depths import ModWrapper
from experiment.models.early_exit import EarlyExitWrapper

from .HasModel import HasModel
from .HasTokenizer import HasTokenizer
from experiment.utils.suppress_output import suppress_all_output


class EvaluationRunner(Runner, HasTokenizer, HasModel):
    """Handles model evaluation, with an optional gating-threshold optimization phase."""

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

    def replace_and_quantize(self, model):
        model_quantized = QuantizedModelForCausalLM.quantize(
            model.model, weights=qint4, exclude="lm_head"
        )

        class QuantizedWrapper(torch.nn.Module):
            def __init__(self, quantized_model):
                super().__init__()
                self.wrapped = quantized_model

            def get_input_embeddings(self):
                return self.wrapped._wrapped.get_input_embeddings()

            def get_output_embeddings(self):
                return self.wrapped._wrapped.get_output_embeddings()

            @property
            def gating(self):
                return self.wrapped._wrapped.gating

            @property
            def model(self):
                return self.wrapped._wrapped

            def generate(self, *args, **kwargs):
                return self.wrapped.generate(*args, **kwargs)

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

            def tie_weights(self):
                self.wrapped._wrapped.tie_weights()

        model.model = QuantizedWrapper(model_quantized)

        print("Quantized model")
        print(model)
        return model

    def _log_percent_tokens_skipped(
        self, model: torch.nn.Module, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        if (
            hasattr(model, "percent_tokens_skipped")
            and len(model.percent_tokens_skipped) != 0
        ):
            results["percent_tokens_skipped"] = sum(model.percent_tokens_skipped) / len(
                model.percent_tokens_skipped
            )

        return results

    def _log_percent_tokens_skipped_per_layer(
        self, model: torch.nn.Module, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        decoder_layers = model.get_decoder_layers(
            model.model
            if not self.evaluation_config.use_quantization
            else model.model.model
        )

        for idx, layer in enumerate(decoder_layers):
            if isinstance(layer, ModWrapper) or isinstance(layer, EarlyExitWrapper):
                layer = layer.module

            mlp = layer.mlp if hasattr(layer, "mlp") else layer.ff
            attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attn

            for module in [mlp, attn]:
                if (
                    hasattr(module, "past_percent_skipped")
                    and len(module.past_percent_skipped) != 0
                ):
                    results[f"percent_tokens_skipped_{module.module_name}_{idx}"] = sum(
                        module.past_percent_skipped
                    ) / len(module.past_percent_skipped)

        return results

    def _log_results(self, model: nn.Module, results: Dict[str, Any], seed: int):
        wandb.init(
            project=self.experiment_config.project_name,
            name=f"{self.experiment_config.experiment_name}_{seed}",
            group=self.experiment_config.experiment_name,
        )
        wandb.log(results)

        # if hasattr(model, "gating_stats_collector"):
        #    with model.gating_stats_collector.visualize_gate_distributions(
        #        model
        #    ) as gate_visualizations:
        #        wandb.log(gate_visualizations)

        wandb.finish()

    def _format_standard_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        print(results.items())
        return {
            f"{key}_{metric}": float(metric_value)
            for key, value in results.items()
            for metric, metric_value in value.items()
            if isinstance(metric_value, numbers.Number)
        }

    def run(self, seed: int, state_dict: torch.Tensor = None) -> Dict[str, float]:
        model = self._load_model(seed, mode="test").to(self.device)
        model.eval()

        if self.evaluation_config.use_quantization:
            model = self.replace_and_quantize(model)

        if state_dict is not None:
            print("Loading state dict for evaluation")
            missing, unexpected = model.load_state_dict(state_dict)
            model = model.to(self.device)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

        model.eval()

        string = self.tokenizer.encode(
            "Joe has 20 horses. He sells 5 of them for $200 each. How much money does he make?",
            return_tensors="pt",
        ).to(self.device)

        print([self.tokenizer.decode(token) for token in string[0]])

        model(string)

        decoder_layers = model.get_decoder_layers(
            model.model
            if not self.evaluation_config.use_quantization
            else model.model.model
        )

        for idx, layer in enumerate(decoder_layers):
            if isinstance(layer, ModWrapper) or isinstance(layer, EarlyExitWrapper):
                layer = layer.module

            mlp = layer.mlp if hasattr(layer, "mlp") else layer.ff
            attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attn

            for module in [mlp, attn]:
                if hasattr(module, "current_token_importance"):
                    print(module.module_name, module.current_token_importance)

        generated = model.generate(
            input_ids=string,
            max_length=100,
            max_new_tokens=100,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        print("Sample generation: ", self.tokenizer.decode(generated[0]))

        # Phase 1: find optimal per-layer thresholds on a small subset
        subset_metric = self.evaluation_config.subset_metric
        # e.g. ['commonsense_qa']
        small_limit = self.evaluation_config.subset_limit
        evaluator_small = ModelEvaluator(
            model,
            self.tokenizer,
            self.evaluation_config.eval_batch_size,
            self.evaluation_config.num_fewshot,
        )

        # Baseline accuracy at t=0
        # with suppress_all_output():
        baseline_out = evaluator_small.evaluate(
            metrics=subset_metric,
            seed=seed,
            experiment_name=f"baseline_{seed}",
            generation_mode=self.model_config.generation_mode,
            limit=self.evaluation_config.subset_limit,
        )
        baseline_acc = baseline_out[subset_metric[0]][
            self.evaluation_config.accuracy_key
        ]

        # Joint eval_fn returning (compute_saved, retention)
        def joint_eval(t: float) -> Tuple[float, float]:
            model.config.skip_threshold = [t]
            with suppress_all_output():
                out = evaluator_small.evaluate(
                    metrics=subset_metric,
                    seed=seed,
                    experiment_name=f"opt_{seed}",
                    generation_mode=self.model_config.generation_mode,
                    limit=self.evaluation_config.subset_limit,
                )
            acc = out[subset_metric[0]][self.evaluation_config.accuracy_key]
            pct_saved = sum(model.percent_tokens_skipped) / len(
                model.percent_tokens_skipped
            )
            return float(pct_saved), float(acc / baseline_acc)

        # Phase 1: optimize thresholds jointly
        opt = ThresholdOptimizer(
            evaluate_fn=joint_eval,
            initial_samples=self.evaluation_config.initial_samples,
            grid_size=500,
        )
        opt.run(iterations=self.evaluation_config.optim_iterations)

        # Example usage: thresholds for 30% compute & 90% retention
        t30 = opt.get_threshold_for_compute(0.30)
        t90 = opt.get_threshold_for_retention(0.90)
        print(f"t@30% compute: {t30:.3f}, t@90% acc: {t90:.3f}")
        # Phase 2: standard evaluation on all metrics with full limit
        evaluator_full = ModelEvaluator(
            model,
            self.tokenizer,
            self.evaluation_config.eval_batch_size,
            self.evaluation_config.num_fewshot,
        )
        metrics = self.evaluation_config.evaluation_metrics
        for compute in tqdm(
            [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            desc="running full eval",
            leave=False,
        ):
            threshold = opt.get_threshold_for_compute(compute)
            print(f"Threshold for compute {compute}: {threshold}")
            self.model_config.skip_threshold = [threshold]
            # run subset evaluation
            with suppress_all_output():
                results = evaluator_full.evaluate(
                    metrics=metrics,
                    seed=seed,
                    experiment_name=f"{self.experiment_config.experiment_name}_thresh_opt_{seed}",
                    generation_mode=self.model_config.generation_mode,
                    limit=self.evaluation_config.full_limit,
                )

            results = self._log_percent_tokens_skipped(model, results)
            results = self._log_percent_tokens_skipped_per_layer(model, results)

            print("Results: ", results)

            if self.experiment_config.enable_logging:
                self._log_results(model, results, seed)

        # Run at 90% accuracy
        with suppress_all_output():
            threshold = opt.get_threshold_for_retention(0.9)

            results = evaluator_full.evaluate(
                metrics=metrics,
                seed=seed,
                experiment_name=f"{self.experiment_config.experiment_name}_thresh_opt_{seed}",
                generation_mode=self.model_config.generation_mode,
                limit=self.evaluation_config.full_limit,
            )

            results = self._log_percent_tokens_skipped(model, results)
            results = self._log_percent_tokens_skipped_per_layer(model, results)

        print(f"Threshold for 90% retetion: {threshold}")
        print("Results: ", results)

        return results
