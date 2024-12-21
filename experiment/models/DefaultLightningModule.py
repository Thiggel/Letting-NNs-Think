import math
from lightning import LightningModule
from transformers import PreTrainedTokenizer
from torch.optim import AdamW
import torch
from typing import Optional
from torch.optim.lr_scheduler import LambdaLR

from experiment.configs import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    UninterruptedMode,
)
from experiment.models.GatedLM import GatedLM

from .model_adapter import ModelAdapter
from .MetricsLogger import MetricsLogger
from .UninterruptedLanguageModel import UninterruptedLanguageModel
from .RecurrentLanguageModel import RecurrentLanguageModel
from .MoDModel import MoDModel
from .HasLayers import HasLayers


class DefaultLightningModule(
    LightningModule,
    UninterruptedLanguageModel,
    RecurrentLanguageModel,
    MoDModel,
    HasLayers,
    GatedLM,
):
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
        self.model_adapter = ModelAdapter(self.config, self.tokenizer, self.device)
        self.model = self.model_adapter.model

        print(self.model)

        self._uninterruted_setup()

        self.metrics_logger = MetricsLogger(
            self, self.tokenizer, self.data_config.batch_size
        )
        self.setup_random_intermediate_supervision()

    def on_before_optimizer_step(self, _):
        """Log gradient norms before optimization step"""
        self.metrics_logger.log_gradient_norms()

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def generate(self, *args, **kwargs):
        self.metrics_logger.dump_first_batch(kwargs)
        return self.model.generate(*args, **kwargs)

    def on_validation_start(self):
        string = self.tokenizer.encode("5 + 10 =", return_tensors="pt").to(self.device)
        generated = self.model.generate(
            input_ids=string, max_length=100, max_new_tokens=100
        )
        print("Sample generation: ", self.tokenizer.decode(generated[0]))

    def lr_lambda_warmup_decay(self, current_step: int) -> float:
        """Get the learning rate for the given step using a lambda function

        The scheduler starts from training_config.initial_lr and scales up to
        training_config.learning_rate during warmup, then decays with cosine schedule
        """
        warmup_steps = self.training_config.warmup_steps
        total_steps = self.training_config.max_training_steps
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
        total_steps = self.training_config.max_training_steps
        min_lr_factor = 0.1

        # Cosine decay from learning_rate to min_lr
        progress = min(1.0, current_step / total_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(min(math.pi, progress * math.pi)))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

    def configure_optimizers(self):
        base_lr = self.training_config.learning_rate
        adam_params = {
            "lr": base_lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.001 if not self.config.enable_normalization else 0.0,
        }

        # Filter trainable parameters for each group
        main_params = [
            p
            for p in self.get_decoder_layers(self.model).parameters()
            if p.requires_grad
        ]

        if self.config.uninterrupted_mode == UninterruptedMode.PROJECTION:
            print("Uninterrupted Projection finetuning")
            main_params += [
                p for p in self.uninterrupted_adapter.parameters() if p.requires_grad
            ]

        embedding_params = [
            p for p in self.model.get_input_embeddings().parameters() if p.requires_grad
        ]

        if self.config.untie_embedding_and_softmax:
            embedding_params += [
                p
                for p in self.model.get_output_embeddings().parameters()
                if p.requires_grad
            ]

        parameters = [
            {
                "params": main_params,
                "lr": base_lr,
            },
            {
                "params": embedding_params,
                "lr": (
                    base_lr / 10 if self.config.untie_embedding_and_softmax else base_lr
                ),
            },
        ]

        # Only create parameter groups if they have parameters
        parameters = [group for group in parameters if len(group["params"]) > 0]

        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(parameters, **adam_params, adamw_mode=True)
        else:
            optimizer = AdamW(parameters, **adam_params)

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=(
                self.lr_lambda_warmup_decay
                if not self.config.enable_normalization
                else self.lr_lambda_decay
            ),
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
        if self.config.enable_normalization:
            self.model_adapter.normalize_weights()

        """Perform a single training/validation/test step"""
        self.metrics_logger.dump_first_batch(batch)

        if self.config.uninterrupted_mode != UninterruptedMode.INTERRUPTED:
            batch["output_hidden_states"] = True
            loss = self.get_recurrent_prediction_loss(batch, mode)

        else:
            outputs = self.model(**batch)
            loss = outputs.loss
            self.metrics_logger.log_metrics(loss, outputs, batch["labels"], mode)

        self.metrics_logger.log_loss(loss, mode)
        loss += self.get_mod_loss()
        loss += self.get_loss_for_intermediate_supervision()
        loss += self.get_gate_loss()

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test")

    def tie_weights(self):
        """Tie the model's weights"""
        if not self.config.untie_embedding_and_softmax:
            self.model.tie_weights()
