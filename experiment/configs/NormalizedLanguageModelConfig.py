from pydantic import Field


class NormalizedLanguageModelConfig:
    enable_normalization: bool = Field(
        False, description="Whether to enable normalization in the model as in nGPT"
    )
