from transformers import AutoTokenizer, PreTrainedTokenizer
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from datasets import Dataset, load_dataset
from pathlib import Path
from pydantic import BaseModel
import os

from experiment.configs import ModelConfig, DataConfig
from experiment.datasets.DatasetConfigurator import DatasetConfigurator


class HasTokenizer:
    configs: dict[str, BaseModel]

    def _initialize_tokenizer(self) -> PreTrainedTokenizer:
        model_config: ModelConfig = self.configs[ModelConfig.__name__]
        data_config: DataConfig = self.configs[DataConfig.__name__]

        # Check if we should use a custom tokenizer
        tokenizer_path = self._get_tokenizer_path(data_config.dataset)

        if tokenizer_path.exists():
            print(f"Loading custom tokenizer from {tokenizer_path}")
            return AutoTokenizer.from_pretrained(str(tokenizer_path))

        if not model_config.pretrained:
            print("Training new tokenizer on dataset")
            return self._train_tokenizer(data_config.dataset)

        # Fall back to pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenizer_path(self, dataset_name: str) -> Path:
        cache_dir = Path(os.getenv("BASE_CACHE_DIR", "."))
        return cache_dir / "tokenizers" / f"{dataset_name}_tokenizer"

    def _train_tokenizer(self, dataset_name: str) -> PreTrainedTokenizer:
        # Initialize a new tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Add normalization and pre-tokenization
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Get dataset config
        dataset_config = DatasetConfigurator.get_dataset_config(dataset_name)

        # Load dataset
        ds = load_dataset(dataset_config["name"], dataset_config.get("subset"))
        train_data = ds[dataset_config["train_field"]]

        # Prepare training data
        def get_training_corpus():
            for i in range(0, len(train_data), 1000):
                batch = train_data[i : i + 1000]
                texts = [
                    dataset_config["q_func"](x) + dataset_config["ans_func"](x)
                    for x in batch
                ]
                yield texts

        # Configure and train the tokenizer
        trainer = trainers.BpeTrainer(
            vocab_size=32000,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

        # Add post-processing
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        # Save the tokenizer
        save_path = self._get_tokenizer_path(dataset_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path / "tokenizer.json"))

        # Convert to PreTrainedTokenizer
        return AutoTokenizer.from_pretrained(str(save_path))
