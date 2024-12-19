from enum import Enum
from typing import Optional, Union, Literal, Annotated
from pydantic import Field

from .LayerRange import LayerRange


class RecurrentMode(str, Enum):
    MAMBA = "mamba"
    TRANSFORMER = "transformer"


class RecursiveTransformerConfig:
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
    add_residual_connection: bool = Field(
        False,
        description="Whether to add residual connections in the model around the recurrent layer",
    )
