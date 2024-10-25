from transformers import AutoTokenizer, PreTrainedTokenizer
from pydantic import BaseModel

from experiment.configs import ModelConfig


class HasTokenizer:
    configs: dict[str, BaseModel]

    def _initialize_tokenizer(self) -> PreTrainedTokenizer:
        model_config: ModelConfig = self.configs[ModelConfig.__name__]

        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
