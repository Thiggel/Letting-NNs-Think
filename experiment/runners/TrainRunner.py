from typing import Any
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import os
from pydantic import BaseModel

from experiment.experiment import Runner
from experiment.datasets import LanguageDataModule
from experiment.model_evaluator import ModelEvaluator
from experiment.models import DefaultLightningModule
from experiment.experiment import ExperimentConfig
from experiment.configs import ModelConfig, DataConfig, TrainingConfig, EvaluationConfig

from .HasTokenizer import HasTokenizer


class TrainRunner(Runner, HasTokenizer):
    """Handles model training using PyTorch Lightning"""

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

    def run(self, seed: int) -> dict[str, float]:
        data_module = LanguageDataModule(
            self.data_config,
            self.model_config,
            self.training_config,
            self.tokenizer,
            seed,
        )
        model = DefaultLightningModule(
            self.model_config, self.training_config, self.data_config, self.tokenizer
        )
        trainer = self._setup_trainer(seed)

        print(model)

        trainer.fit(model=model, datamodule=data_module)

        if (
            self.evaluation_config.save_to_checkpoint
            and trainer.checkpoint_callback
            and trainer.checkpoint_callback.best_model_path
        ):
            self._save_checkpoint(trainer.checkpoint_callback.best_model_path, seed)

        return {}

    def _setup_trainer(self, seed: int) -> Trainer:
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                dirpath=(
                    os.environ.get("PYTORCH_LIGHTNING_HOME")
                    if torch.cuda.is_available()
                    else None
                ),
                filename=self.experiment_config.experiment_name
                + "_"
                + str(seed)
                + "_best-checkpoint-{epoch:02d}-{val_loss:.5f}",
            ),
            DeviceStatsMonitor(),
            EarlyStopping(
                monitor="val_loss", patience=1, mode="min", min_delta=0.00, verbose=True
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

        trainer_args = self._get_trainer_args(callbacks, seed)

        if torch.cuda.is_available():
            trainer_args.update(self._get_cuda_specific_args())

        return Trainer(**trainer_args)

    def _get_trainer_args(self, callbacks: list, seed: int) -> dict[str, Any]:
        wandb_logger = None
        if self.experiment_config.enable_logging:
            wandb_logger = WandbLogger(
                project="variable-depth-lms3",
                name=f"{self.experiment_config.experiment_name}_{seed}",
                group=self.experiment_config.experiment_name,
                save_dir=os.environ["WANDB_DIR"],
                log_model="all",
            )

        grad_acc_steps = (
            1
            if self.data_config.batch_size >= 64
            else 64 // self.data_config.batch_size
        )

        return {
            "callbacks": callbacks,
            "enable_checkpointing": True,
            "logger": wandb_logger if self.experiment_config.enable_logging else None,
            "log_every_n_steps": 10,
            "max_epochs": self.training_config.max_epochs,
            "max_time": {"days": 4, "hours": 12},
            "gradient_clip_val": self.training_config.max_grad_norm,
            "accumulate_grad_batches": grad_acc_steps,
            "devices": "auto",
        }

    def _get_cuda_specific_args(self) -> dict[str, Any]:
        return {
            "strategy": "deepspeed_stage_3_offload",
            "precision": "bf16",
            "default_root_dir": os.environ["PYTORCH_LIGHTNING_HOME"],
        }

    def _save_checkpoint(self, checkpoint_path: str, seed):
        output_path = os.path.join(
            os.environ["BASE_CACHE_DIR"],
            f"{self.evaluation_config.save_to_checkpoint}_{seed}.pt",
        )
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, output_path)
