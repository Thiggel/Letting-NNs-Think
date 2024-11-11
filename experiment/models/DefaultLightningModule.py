import math
from lightning import LightningModule
from transformers import PreTrainedTokenizer
from torch.optim import AdamW
import torch
from typing import Optional
from torch.optim.lr_scheduler import LambdaLR

from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import ModelConfig, TrainingConfig

from .ModelAdapter import ModelAdapter
from .MetricsLogger import MetricsLogger


class DefaultLightningModule(LightningModule):
    """Main Lightning Module for language model training"""

    def __init__(
        self,
        config: ModelConfig,
        training_config: TrainingConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_adapter: Optional[ModelAdapter] = None,
    ):
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.tokenizer = tokenizer

        # Initialize components
        if model_adapter is not None:
            self.model_adapter = model_adapter
            self.model = model_adapter.model
        else:
            self.model_adapter = ModelAdapter(config)
            self.model = self.model_adapter.model

        self.metrics_logger = MetricsLogger(self)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

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

        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(
                self.parameters(), **adam_params, adamw_mode=True
            )
        else:
            optimizer = AdamW(self.parameters(), **adam_params)

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

    def check_for_nans(self) -> bool:
        """Check for NaN values in model parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                print(f"Found NaN in {name}")
                return True
        return False

    def on_validation_epoch_end(self):
        questions = [
            "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?",
            "Farmer Brown has 20 animals on his farm, all either chickens or cows. They have a total of 70 legs, all together. How many of the animals are chickens?",
        ]

        for question in questions:
            input_ids = self.tokenizer.encode(question, return_tensors="pt").cuda()
            output = self.model.generate(input_ids, max_length=200)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(decoded_output)
            print()

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
