import math
from lightning import LightningModule
from transformers import PreTrainedTokenizer
from torch.optim import AdamW
import torch
import torch.nn.functional as F
from typing import Optional
from torch.optim.lr_scheduler import LambdaLR

from experiment.configs import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
)
from experiment.configs.ModelConfig import FinetuneMode

from .model_adapter import ModelAdapter
from .MetricsLogger import MetricsLogger
from .HasLayers import HasLayers

from deepspeed.utils import safe_get_full_grad


class DefaultLightningModule(LightningModule, HasLayers):
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

        self.model_adapter = ModelAdapter(self.config, self.tokenizer, self.device)
        self.model = self.model_adapter.model
        self.old_forward = self.model.forward
        self.model.forward = self.forward
        self.percent_tokens_skipped = []

        self.metrics_logger = MetricsLogger(
            self, self.tokenizer, self.data_config.batch_size
        )

    def on_before_optimizer_step(self, _):
        """Log gradient norms before optimization step"""
        capacity = self.config.mod_capacity_factor + (
            1 - self.config.mod_capacity_factor
        ) * (1.0 / (self.global_step / 1000 + 1))

        if self.config.use_mod:
            self.model.mod.update_capacity(capacity)

            self.log("mod_capacity", capacity, sync_dist=True)

        # self.metrics_logger.log_gradient_norms()

    def forward(self, input_ids, **kwargs):
        output = self.old_forward(input_ids, **kwargs)

        if self.config.use_gating:
            for name, module in self.model.gating.wrapped_modules.items():
                self.percent_tokens_skipped.append(
                    module.current_percent_tokens_skipped
                )
        elif self.config.use_mod:
            for module in self.model.mod.wrapped_modules.values():
                self.percent_tokens_skipped.append(
                    1 - module.current_percent_tokens_processed
                )

        return output

    def generate(self, *args, **kwargs):
        self.metrics_logger.dump_first_batch(kwargs)
        output = self.model.generate(*args, **kwargs)

        print(
            "Output: ",
            self.tokenizer.decode(output[0]),
        )

        print("Tokens skipped: ", torch.mean(torch.tensor(self.percent_tokens_skipped)))

        return output

    def sample_generate(self):
        string = self.tokenizer.encode(
            "My dog is ",
            return_tensors="pt",
        ).to(self.device)

        generated = self.model.generate(
            input_ids=string,
            max_length=100,
            max_new_tokens=100,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        print("Sample generation: ", self.tokenizer.decode(generated[0]))

    def on_validation_start(self):
        self.sample_generate()

    def lr_lambda_warmup_decay(self, current_step: int) -> float:
        """Get the learning rate for the given step using a lambda function

        The scheduler starts from training_config.initial_lr and scales up to
        training_config.learning_rate during warmup, then decays with cosine schedule
        """
        warmup_steps = self.training_config.warmup_steps
        default_max_steps = 10_000
        total_steps = self.training_config.max_training_steps or default_max_steps
        min_lr_factor = 0.1

        # Calculate the ratio between initial and target learning rate
        lr_ratio = self.training_config.initial_lr / self.training_config.learning_rate

        if current_step < warmup_steps:
            # Linear warmup from initial_lr to learning_rate
            warmup_factor = float(current_step) / float(max(1, warmup_steps))
            return lr_ratio + (1.0 - lr_ratio) * warmup_factor
        else:
            # Cosine decay from learning_rate to min_lr
            progress = min(
                1.0, (current_step - warmup_steps) / (total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(min(math.pi, progress * math.pi)))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

    def lr_lambda_decay(self, current_step: int) -> float:
        """Get the learning rate for the given step using a lambda function

        The scheduler starts from training_config.initial_lr and decays with cosine schedule
        """
        default_max_steps = 10_000
        total_steps = self.training_config.max_training_steps or default_max_steps
        min_lr_factor = 0.1

        # Cosine decay from learning_rate to min_lr
        progress = min(1.0, current_step / total_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(min(math.pi, progress * math.pi)))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

    def configure_optimizers(self):
        base_lr = self.training_config.learning_rate
        adam_params = {
            "betas": (0.9, 0.95),
            "weight_decay": 0.001,
        }

        model_params = [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and "router" not in name and "predictor" not in name
        ]

        mod_params = [
            param
            for name, param in self.model.named_parameters()
            if "router" in name or "predictor" in name
        ]

        parameters = [
            {"params": model_params, "lr": base_lr},
            {"params": mod_params, "lr": 1e-3},
        ]

        if torch.cuda.is_available() and self.training_config.use_deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(parameters, **adam_params, adamw_mode=True)
        else:
            optimizer = AdamW(parameters, **adam_params)

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=self.lr_lambda_warmup_decay,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _step(self, batch, _: int, mode: str = "train") -> torch.Tensor:
        """Perform a single training/validation/test step"""
        self.metrics_logger.dump_first_batch(batch)

        outputs = self.model(**batch)

        if self.config.use_kl_div_training:
            logits = outputs.logits

            self.model_adapter.orig_model.to(self.device)
            ref_logits = self.model_adapter.orig_model(**batch).logits

            loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )

            self.log(
                f"{mode}_kl_div_loss",
                loss,
                sync_dist=True,
                batch_size=batch["labels"].shape[0],
            )

            self.log(
                f"{mode}_cross_entropy_loss",
                outputs.loss,
                sync_dist=True,
                batch_size=batch["labels"].shape[0],
            )
            self.metrics_logger.log_metrics(
                outputs.loss, outputs, batch["labels"], mode
            )

        else:
            loss = outputs.loss

            self.log(
                f"{mode}_cross_entropy_loss",
                loss,
                sync_dist=True,
                batch_size=batch["labels"].shape[0],
            )

            self.metrics_logger.log_metrics(loss, outputs, batch["labels"], mode)

        if self.config.use_gating:
            gate_entropy_loss, gate_sparsity_loss = (
                self.model.gating.compute_gate_loss()
            )

            self.log(
                f"{mode}_gate_sparsity_loss",
                gate_sparsity_loss,
                sync_dist=True,
                batch_size=batch["labels"].shape[0],
            )

            loss += gate_entropy_loss * self.config.entropy_loss_weight
            loss += gate_sparsity_loss * self.config.sparsity_loss_weight

        elif self.config.use_mod:
            predictor_loss = self.model.mod.compute_predictor_loss(dtype=loss.dtype)

            self.log(
                f"{mode}_predictor_loss",
                predictor_loss,
                sync_dist=True,
                batch_size=batch["labels"].shape[0],
            )

            loss += predictor_loss * self.config.predictor_loss_weight

        self.metrics_logger.log_loss(loss, mode)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        self.sample_generate()
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test")

    def tie_weights(self):
        """Tie the model's weights"""
        self.model.tie_weights()
