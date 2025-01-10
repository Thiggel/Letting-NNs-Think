from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation and checkpointing"""

    evaluation_metrics: Optional[list[str]] = Field(
        None, description="The evaluation metrics to use"
    )
    load_from_checkpoint: Optional[Path] = Field(
        None, description="The path to the checkpoint to load"
    )
    save_to_checkpoint: Optional[Path] = Field(
        None, description="The path to the checkpoint to save to"
    )
    evaluate_as_uninterrupted: bool = Field(
        False, description="Generate using the uninterrupted technique"
    )
    eval_batch_size: int = Field(
        128, description="The batch size to use for evaluation"
    )
    num_fewshot: int = Field(0, description="The number of few-shot examples to use")
    uninterrupted_alpha: float = Field(
        0.0, description="The alpha value to use for uninterrupted generation"
    )
