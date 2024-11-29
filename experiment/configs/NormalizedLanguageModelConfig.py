from pydantic import Field


class NormalizedLanguageModelConfig:
    enable_normalization: bool = Field(
        False, description="Whether to enable normalization in the model as in nGPT"
    )
    use_dynamic_eigen_lrs: bool = Field(
        True, description="Whether to use dynamic eigen learning rates"
    )
    use_momentum: bool = Field(
        False, description="Whether to use momentum in the model"
    )
