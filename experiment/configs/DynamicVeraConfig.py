from pydantic import Field


class DynamicVeraConfig:
    use_dynamic_vera: bool = Field(
        False, description="Whether to use the Dynamic VeRA layer"
    )
    vera_r: int = Field(8, description="The inner size of the low-rank matrix in VeRA")
