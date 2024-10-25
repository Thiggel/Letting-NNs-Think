from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation and checkpointing"""

    evaluate: bool = Field(True, description="Whether to evaluate the model")
    evaluation_metrics: Optional[list[str]] = Field(
        None, description="The evaluation metrics to use"
    )
    load_from_checkpoint: Optional[Path] = Field(
        None, description="The path to the checkpoint to load"
    )
    save_to_checkpoint: Optional[Path] = Field(
        None, description="The path to the checkpoint to save to"
    )
