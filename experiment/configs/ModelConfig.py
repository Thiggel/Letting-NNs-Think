from enum import Enum
from typing import Optional, Union, Literal
from pydantic import BaseModel, Field


class RecurrentMode(str, Enum):
    MAMBA = "mamba"
    TRANSFORMER = "transformer"


class LayerIndex:
    def __init__(self, value: Union[int, tuple[int, int]]):
        self.value = value

    @classmethod
    def validate(cls, value: Union[int, str]) -> "LayerIndex":
        if isinstance(value, int):
            return cls(value)
        try:
            if ":" in value:
                start, end = map(int, value.split(":"))
                return cls((start, end))
            return cls(int(value))
        except ValueError:
            raise ValueError("Must be an int or two ints separated by a colon")


class NumSteps:
    def __init__(
        self, value: Union[int, Literal["classifier", "fixed_point", "random"]]
    ):
        self.value = value

    @classmethod
    def validate(cls, value: Union[int, str]) -> "NumSteps":
        if isinstance(value, int):
            return cls(value)
        if value in {"classifier", "fixed_point", "random"}:
            return cls(value)
        try:
            return cls(int(value))
        except ValueError:
            raise ValueError(
                "Must be an int or one of ['classifier', 'fixed_point', 'random']"
            )


class ModelConfig(BaseModel):
    """Configuration for model architecture and behavior"""

    model_name: str = Field("gpt2", description="The model name to be used")
    finetune_layers: Optional[Union[Literal["all"], list[int]]] = Field(
        None, description="The layers to fine-tune"
    )
    make_layers_recurrent: Optional[LayerIndex] = Field(
        None, description="The layer to make recurrent"
    )
    recurrent_mode: RecurrentMode = Field(
        RecurrentMode.TRANSFORMER, description="The recurrent mode to use"
    )
    num_steps: Optional[NumSteps] = Field(
        None,
        description="Number of steps as an integer or 'classifier'/'fixed_point'/'random'",
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
