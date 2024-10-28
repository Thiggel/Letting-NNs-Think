from lightning import LightningModule
from transformers import PreTrainedTokenizer
from torch.optim import AdamW
import torch
from typing import Optional
from torch.optim.lr_scheduler import LambdaLR

from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import ModelConfig

from .ModelAdapter import ModelAdapter
from .MetricsLogger import MetricsLogger


class DefaultLightningModule(LightningModule):
    """Main Lightning Module for language model training"""

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__()
        self.config = config
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
        # Choose optimizer based on GPU availability
        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=1e-4, betas=(0.9, 0.95))
        else:
            optimizer = AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95))

        # Define the number of warmup steps (e.g., 10% of total training steps)
        warmup_steps = 200  # Adjust this value based on your training setup
        total_steps = 2000  # Total number of training steps (adjust as needed)

        # Create a lambda function for linear warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # After warmup, keep the learning rate constant

        # Create the scheduler with the lambda function
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # Return optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or 'epoch' if you prefer to update per epoch
                "frequency": 1,  # How often to update the learning rate (1 means every step/epoch)
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

        # Log metrics
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
