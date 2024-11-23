import math
from lightning import LightningModule
from transformers import PreTrainedTokenizer
from torch.optim import AdamW
import torch
import torch.nn.functional as F
from typing import Optional
from torch.optim.lr_scheduler import LambdaLR

from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import ModelConfig, TrainingConfig, DataConfig

from .ModelAdapter import ModelAdapter
from .MetricsLogger import MetricsLogger
from .UninterruptedLanguageModel import UninterruptedLanguageModel


class DefaultLightningModule(LightningModule, UninterruptedLanguageModel):
    """Main Lightning Module for language model training"""

    def __init__(
        self,
        config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.data_config = data_config
        self.tokenizer = tokenizer

    def setup(self, stage):
        self.model_adapter = ModelAdapter(self.config, self.device)
        self.model = self.model_adapter.model

        print(self.model)

        self.metrics_logger = MetricsLogger(self, self.data_config.batch_size)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def lr_lambda(self, current_step: int) -> float:
        """Get the learning rate for the given step using a lambda function"""
        warmup_steps = self.training_config.warmup_steps
        total_steps = self.training_config.total_training_steps
        min_lr_factor = 0.1

        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = min(
                1.0, (current_step - warmup_steps) / (total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(min(math.pi, progress * math.pi)))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

    def configure_optimizers(self):
        adam_params = {
            "lr": self.training_config.learning_rate,
            "betas": (0.9, 0.95),
            "weight_decay": 0.001,
        }

        parameters = [
            {
                "params": self.model.base_model.model.model.layers.parameters(),
                "lr": self.training_config.learning_rate,
            },
        ]

        if self.config.finetune_mode in ["lastlayer_lmhead", "lmhead_lora"]:
            parameters.append(
                {
                    "params": self.model.base_model.model.lm_head.parameters(),
                    "lr": self.training_config.learning_rate / 100,
                }
            )

        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(parameters, **adam_params, adamw_mode=True)
        else:
            optimizer = AdamW(parameters, **adam_params)

        scheduler = LambdaLR(optimizer, lr_lambda=self.lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimization step"""
        self.metrics_logger.log_gradient_norms()

    def get_recurrent_layer(self) -> Optional[RecurrentTransformerLayer]:
        """Get the recurrent layer if it exists"""
        if not hasattr(self.model_adapter, "recurrent_layer_idx"):
            return None
        return self.model.model.layers[self.model_adapter.recurrent_layer_idx]

    def get_loss_for_intermediate_supervision(self) -> torch.Tensor:
        layer = self.get_recurrent_layer()

        if (
            not self.training_config.use_random_intermediate_supervision
            or layer is None
            or len(layer.intermediate_outputs) == 0
        ):
            return 0

        intermediate_outputs = torch.stack(layer.intermediate_outputs)

        loss = F.mse_loss(
            intermediate_outputs,
            torch.randn_like(intermediate_outputs),
        )

        return loss

    def get_mod_loss(self) -> torch.Tensor:
        """Get the MoD loss"""
        if not hasattr(self.model, "mod_loss"):
            return 0

        self.log("mod_loss", self.model.mod_loss)

        return self.model.mod_loss

    def _dump_first_batch(self, batch: dict[str, torch.Tensor]) -> None:
        if self.trainer.global_rank > 0:
            return

        MAX_DUMPS = 5

        if not hasattr(self, "num_dumped_first_batch"):
            self.num_dumped_first_batch = 0

        if self.num_dumped_first_batch < MAX_DUMPS:
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

    def _step(self, batch, _: int, mode: str = "train") -> torch.Tensor:
        """Perform a single training/validation/test step"""
        self._dump_first_batch(batch)

        if self.config.make_uninterrupted:
            batch["output_hidden_states"] = True

        if self.config.make_uninterrupted:
            loss = self.get_recurrent_prediction_loss(batch, mode)

        else:
            outputs = self.model(**batch)
            loss = outputs.loss
            self.metrics_logger.log_metrics(loss, outputs, batch["labels"], mode)

        self.metrics_logger.log_loss(loss, mode)
        loss += self.get_mod_loss()
        loss += self.get_loss_for_intermediate_supervision()

        current_weights = self.model.base_model.model.lm_head.weight.data
        print(current_weights)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test")

    def tie_weights(self):
        """Tie the model's weights"""
        self.model.tie_weights()

    def track_weight_changes(self, layer_name=None):
        """
        Track changes in model weights.
        Args:
            layer_name: Optional specific layer to track. If None, tracks all layers.
        Returns:
            dict: Changes in weights with their magnitudes
        """
        changes = {}
        from copy import deepcopy

        for name, param in self.named_parameters():
            if layer_name is not None and name != layer_name:
                continue

            if name not in self.previous_weights:
                self.previous_weights[name] = deepcopy(param.data)
                changes[name] = {"changed": False, "max_diff": 0.0, "mean_diff": 0.0}
            else:
                diff = param.data - self.previous_weights[name]
                max_diff = torch.max(torch.abs(diff)).item()
                mean_diff = torch.mean(torch.abs(diff)).item()

                changes[name] = {
                    "changed": not torch.allclose(
                        param.data, self.previous_weights[name]
                    ),
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                }

                # Update stored weights
                self.previous_weights[name] = deepcopy(param.data)

        return changes

    def on_after_backward(self):
        # Track changes after backward pass
        post_backward_changes = self.track_weight_changes()
        for name, change_info in post_backward_changes.items():
            if change_info["changed"]:
                self.log(f"{name}_max_diff", change_info["max_diff"])
                self.log(f"{name}_mean_diff", change_info["mean_diff"])
