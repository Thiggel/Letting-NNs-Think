from typing import Any, Dict
import wandb
import torch
from pydantic import BaseModel
from optimum.quanto import QuantizedModelForCausalLM, qint4

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

    def run(self, seed: int, state_dict: torch.Tensor = None) -> Dict[str, float]:
        model = self._load_model(seed, mode="test")
        model.to(self.device)

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

        # Determine if we're evaluating on synthetic datasets
        synthetic_tasks = ["arithmetic", "pattern", "complex_arithmetic_reasoning"]
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
                self.evaluation_config.eval_batch_size,
                self.evaluation_config.num_fewshot,
            )
            standard_results = evaluator.evaluate(
                standard_metrics,
                seed,
                self.experiment_config.experiment_name,
                self.model_config.generation_mode,
            )
            results.update(self._format_standard_results(standard_results))

        results = self._log_percent_tokens_skipped(model, results)
        results = self._log_percent_tokens_skipped_per_layer(model, results)

        if self.experiment_config.enable_logging:
            self._log_results(results, seed)

        return results

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
