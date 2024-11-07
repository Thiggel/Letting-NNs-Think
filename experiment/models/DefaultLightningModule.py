import math
from lightning import LightningModule
from transformers import PreTrainedTokenizer
from torch.optim import AdamW
import torch
from typing import Optional
from torch.optim.lr_scheduler import LambdaLR

from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import ModelConfig, TrainingConfig, DataConfig

from .ModelAdapter import ModelAdapter
from .MetricsLogger import MetricsLogger


class DefaultLightningModule(LightningModule):
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

        # Initialize components
        self.model_adapter = ModelAdapter(config)
        self.model = self.model_adapter.model
        self.metrics_logger = MetricsLogger(self)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def configure_optimizers(self):
        adam_params = {
            "lr": self.config.learning_rate,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
        }
        # Choose optimizer based on GPU availability
        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(
                self.parameters(), **adam_params, adamw_mode=True
            )
        else:
            optimizer = AdamW(self.parameters(), **adam_params)

        # Define the number of warmup steps and total steps
        total_steps = 1000
        warmup_steps = total_steps // 10
        min_lr_factor = 0.1  # Final learning rate will be 10% of max

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # One-way cosine decay after warmup
                progress = min(
                    1.0, (current_step - warmup_steps) / (total_steps - warmup_steps)
                )
                # Only use the first half of the cosine curve (from 0 to π)
                cosine_decay = 0.5 * (1.0 + math.cos(min(math.pi, progress * math.pi)))
                # Scale the decay to range from 1.0 to min_lr_factor
                return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

        # Create the scheduler with the lambda function
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

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

    def check_for_nans(self) -> bool:
        """Check for NaN values in model parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                print(f"Found NaN in {name}")
                return True
        return False

    def get_recurrent_layer(self) -> Optional[RecurrentTransformerLayer]:
        """Get the recurrent layer if it exists"""
        if not hasattr(self.model_adapter, "recurrent_layer_idx"):
            return None
        return self.model.model.layers[self.model_adapter.recurrent_layer_idx]

    def _step(self, batch, batch_idx, mode: str = "train") -> torch.Tensor:
        """Perform a single training/validation/test step"""
        outputs = self.model(**batch)
        loss = outputs.loss

        self.metrics_logger.log_loss(loss, mode)
        self.metrics_logger.log_metrics(loss, outputs, batch["labels"], mode)

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
