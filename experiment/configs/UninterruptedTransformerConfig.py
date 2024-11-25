from pydantic import Field


class UninterruptedTransformerConfig:
    make_uninterrupted: bool = Field(
        False,
        description="Whether to make the model uninterrupted by making the last hidden state similar to the next token's first embedded state",
    )
    uninterrupted_loss_weight: float = Field(
        1.0, description="The weight for the uninterrupted loss"
    )
    uninterrupted_recurrence_depth: int = Field(
        5, description="The depth of recurrence"
    )
    recurrent_prediction_weight: float = Field(
        1.0, description="The weight for the recurrent prediction loss"
    )
    recurrent_hidden_state_weight: float = Field(
        1.0, description="The weight for the recurrent hidden state loss"
    )
    untie_embedding_and_softmax: bool = Field(
        False,
        description="Whether to untie the embedding and softmax weights",
    )
