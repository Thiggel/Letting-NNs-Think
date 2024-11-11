from enum import Enum
import argparse
from typing import Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field


class LayerRange:
    """Custom type for layer ranges that can be a single number or range"""

    @staticmethod
    def parse(value: str) -> Union[int, tuple[int, int]]:
        if ":" in value:
            try:
                start, end = map(int, value.split(":"))
                if start >= end:
                    raise ValueError("Start must be less than end")
                return (start, end)
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"Invalid range format: {e}")
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer: {value}")


class RecurrentMode(str, Enum):
    MAMBA = "mamba"
    TRANSFORMER = "transformer"


class FinetuneMode(str, Enum):
    FULL = "full"
    LORA = "lora"


class ModelConfig(BaseModel):
    """Configuration for model architecture and behavior"""

    model_name: str = Field("gpt2", description="The model name to be used")
    make_layers_recurrent: Optional[Annotated[str, LayerRange]] = Field(
        None, description="The layers to make recurrent (e.g., '5' or '2:4')"
    )
    recurrent_mode: RecurrentMode = Field(
        "transformer", description="The recurrent mode to use"
    )
    num_steps: Optional[Union[int, Literal["classifier", "fixed_point", "random"]]] = (
        Field(
            None,
            description="Number of steps as an integer or 'classifier'/'fixed_point'/'random'",
        )
    )
    max_steps: int = Field(20, description="The maximum number of steps to take")
    use_exit_tokens: bool = Field(
        False,
        description="Whether to use exit tokens in the model or let the classifier operate on the hidden states of the normal tokens",
    )
    exit_threshold: float = Field(
        0.5,
        description="The threshold for the exit classifier to determine whether to stop early",
    )
    use_time_embedding: bool = Field(
        False, description="Whether to use a time embedding in the model"
    )
    use_gating: bool = Field(False, description="Whether to use a gating in the model")
    finetune_mode: FinetuneMode = Field(
        "full", description="The finetuning mode to use"
    )
    lora_r: int = Field(8, description="The LoRA rank")
    lora_alpha: int = Field(1, description="The LoRA alpha")
    lora_target_modules: list[str] = Field(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="The target modules for LoRA",
    )
    lora_dropout: float = Field(0.0, description="The LoRA dropout rate")
