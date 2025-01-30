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
)
from experiment.configs.ModelConfig import FinetuneMode

from .model_adapter import ModelAdapter
from .MetricsLogger import MetricsLogger
from .HasLayers import HasLayers


class DefaultLightningModule(
    LightningModule,
    HasLayers,
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

        self.model_adapter = ModelAdapter(self.config, self.tokenizer, self.device)
        self.model = self.model_adapter.model

        print(self.model)

        self.metrics_logger = MetricsLogger(
            self, self.tokenizer, self.data_config.batch_size
        )

    def on_before_optimizer_step(self, _):
        """Log gradient norms before optimization step"""
        # self.metrics_logger.log_gradient_norms()

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def generate(self, *args, **kwargs):
        self.metrics_logger.dump_first_batch(kwargs)
        return self.model.generate(*args, **kwargs)

    def sample_generate(self):
        string = self.tokenizer.encode(
            "query : 5 + 0 * 9 - 5 + 1 + 4 * 9 + 4 - 3 + 6 + 9 answer :",
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
            "lr": base_lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.001,
        }

        main_params = []
        if self.config.finetune_mode != FinetuneMode.FROZEN:
            main_params = [p for p in self.model.parameters() if p.requires_grad]

        parameters = [
            {
                "params": main_params,
                "lr": base_lr,
            },
        ]

        # Only create parameter groups if they have parameters
        parameters = [group for group in parameters if len(group["params"]) > 0]

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
        loss = outputs.loss
        self.metrics_logger.log_metrics(loss, outputs, batch["labels"], mode)

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
        if not self.config.untie_embedding_and_softmax:
            self.model.tie_weights()
