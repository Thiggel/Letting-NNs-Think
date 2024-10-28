from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for dataset and data processing"""

    dataset: str = Field("gsm8k", description="The dataset to use for training")
    seq_length: int = Field(1024, description="The maximum sequence length")
    batch_size: int = Field(8, description="The batch size")
