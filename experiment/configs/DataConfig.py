from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for dataset and data processing"""

    dataset: str = Field(
        ..., description="The dataset to use for training"
    )  # Required field
    seq_length: int = Field(1024, description="The maximum sequence length")
