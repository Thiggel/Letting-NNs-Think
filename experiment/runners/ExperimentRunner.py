from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import wandb
import os
from abc import ABC, abstractmethod

from experiment.experiment import Runner


class ExperimentRunner(Runner):
    """Main runner that delegates to specific runners based on mode"""

    def __init__(self, configs: dict[str, BaseModel], mode: str):
        self.configs = configs
        self.mode = mode
        self.runner = self._create_runner()

    def _create_runner(self) -> BaseRunner:
        if self.mode == "train":
            return TrainRunner(self.configs)
        elif self.mode == "evaluate":
            return EvaluationRunner(self.configs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def set_seed(seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self, seed: int) -> Dict[str, float]:
        self.set_seed(seed)
        return self.runner.run(seed)
