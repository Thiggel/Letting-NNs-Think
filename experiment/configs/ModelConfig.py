from enum import Enum
from typing import Optional, Annotated
from pydantic import BaseModel, Field

from .RecursiveTransformerConfig import RecursiveTransformerConfig
from .UninterruptedTransformerConfig import UninterruptedTransformerConfig
from .LoraConfig import LoraConfig
from .DynamicVeraConfig import DynamicVeraConfig
from .MixtureOfDepthsConfig import MixtureOfDepthsConfig
from .NormalizedLanguageModelConfig import NormalizedLanguageModelConfig
from .LayerRange import LayerRange


class FinetuneMode(str, Enum):
    FULL = "full"
    LORA = "lora"
    UNINTERRUPTED = "lastlayer_lmhead"
    UNINTERRUPTED_LORA = "lmhead_lora"
    FROZEN = "frozen"


class ModelConfig(
    BaseModel,
    RecursiveTransformerConfig,
    UninterruptedTransformerConfig,
    LoraConfig,
    DynamicVeraConfig,
    MixtureOfDepthsConfig,
    NormalizedLanguageModelConfig,
):
    """Configuration for model architecture and behavior"""

    model_name: str = Field("gpt2", description="The model name to be used")
    pretrained: bool = Field(
        False, description="Whether to use a pretrained model or not"
    )
    finetune_mode: FinetuneMode = Field(
        "lora", description="The finetuning mode to use"
    )
    remove_layers: Optional[Annotated[str, LayerRange]] = Field(
        None, description="The layers to remove from the model"
    )
