from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation and checkpointing"""

    evaluation_metrics: Optional[list[str]] = Field(
        None, description="The evaluation metrics to use"
    )
    subset_metric: str = Field(
        "csqa_gen", description="The metric to use for subset evaluation"
    )
    load_from_checkpoint: Optional[Path] = Field(
        None, description="The path to the checkpoint to load"
    )
    save_to_checkpoint: Optional[Path] = Field(
        None, description="The path to the checkpoint to save to"
    )
    eval_batch_size: int = Field(
        128, description="The batch size to use for evaluation"
    )
    num_fewshot: int = Field(0, description="The number of few-shot examples to use")
    use_quantization: bool = Field(
        False, description="Whether to use quantization for evaluation"
    )
