from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
import torch


class PPOState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: torch.Tensor = Field(
        ...,
        description="The state of the environment. E.g. the hidden state at a certain step of the LM.",
    )
    action: torch.Tensor = Field(
        ..., description="The action taken in the environment. E.g. exit or continue."
    )
    action_log_prob: torch.Tensor = Field(
        ..., description="The log probability of the action taken."
    )
    entropy: torch.Tensor = Field(
        ..., description="The entropy of the probability distribution."
    )
    reward: Optional[torch.Tensor] = Field(
        None, description="The reward received from the environment."
    )
    advantage: Optional[torch.Tensor] = Field(
        None, description="The advantage of the action taken."
    )
    next_state: Optional[torch.Tensor] = Field(
        None, description="The next state of the environment."
    )
