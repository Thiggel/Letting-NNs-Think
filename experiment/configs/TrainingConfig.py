from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TrainingProcedureType(str, Enum):
    DEFAULT = "default"
    REINFORCE = "reinforce"
    UNINTERRUPTED = "uninterrupted"
    MAKE_UNINTERRUPTED = "make_uninterrupted"


class TrainingConfig(BaseModel):
    """Configuration for training parameters"""

    max_epochs: int = Field(5, description="The maximum number of epochs")
    warmup_steps: int = Field(1000, description="The number of warmup steps")
    max_grad_norm: float = Field(
        1.0, description="The maximum gradient norm for gradient clipping"
    )
    training_procedure: Optional[TrainingProcedureType] = Field(
        None, description="The training procedure to use"
    )
    gamma: float = Field(0.99, description="The discount factor")
    temperature: float = Field(1.0, description="The sampling temperature")
    baseline_decay: float = Field(0.99, description="The decay factor for the baseline")
    use_random_intermediate_supervision: bool = Field(
        False, description="Whether to use random intermediate supervision"
    )
