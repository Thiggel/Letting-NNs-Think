from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    experiment_name: str = Field("default", description="Name of the experiment")
    seeds: list[int] = Field([1, 2, 3], description="Random seeds to use")
    num_runs: int = Field(3, description="Number of runs to perform")
    enable_logging: bool = Field(True, description="Enable wandb logging")
    continue_on_error: bool = Field(False, description="Continue if a run fails")
    save_results: bool = Field(True, description="Save results to disk")
