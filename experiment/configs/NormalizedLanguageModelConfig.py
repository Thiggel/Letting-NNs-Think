from pydantic import Field


class NormalizedLanguageModelConfig:
    enable_normalization: bool = Field(
        False, description="Whether to enable normalization in the model as in nGPT"
    )
    use_dynamic_eigen_lrs: bool = Field(
        True, description="Whether to use dynamic eigen learning rates"
    )
    use_lr_projection: bool = Field(
        False, descripion="Whether to use gating in the model"
    )
    use_momentum: bool = Field(
        False, description="Whether to use momentum in the model"
    )
