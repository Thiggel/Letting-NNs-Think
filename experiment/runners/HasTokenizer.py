from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    processors,
)
from pydantic import BaseModel
from pathlib import Path
import os
from datasets import load_dataset

from experiment.datasets import DatasetConfigurator
from experiment.configs import ModelConfig, DataConfig


class HasTokenizer:
    configs: dict[str, BaseModel]

    def _get_tokenizer_path(self, dataset_name: str) -> Path:
        """Get the path where tokenizers should be stored."""
        return Path(
            os.getenv("BASE_CACHE_DIR", "") + f"/tokenizers/{dataset_name}_tokenizer"
        )

    def _initialize_tokenizer(self) -> PreTrainedTokenizer:
        model_config: ModelConfig = self.configs[ModelConfig.__name__]
        data_config: DataConfig = self.configs[DataConfig.__name__]

        tokenizer_path = self._get_tokenizer_path(data_config.dataset)

        if (tokenizer_path / "tokenizer.json").exists():
            print(f"Loading custom tokenizer from {tokenizer_path}")

            fast_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_path / "tokenizer.json"),
                bos_token="[BOS]",
                eos_token="[EOS]",
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
            )

            print("\nExample tokenizations:")
            test_strings = ["1 * 9 + 8 + 4 * 18 + 19 - 2 + 3 + 8 = 117"]
            for test in test_strings:
                ids = fast_tokenizer.encode(str(test))
                decoded = fast_tokenizer.decode(ids)
                print(f"'{test}' -> {ids} -> '{decoded}'")

            # Verify special token IDs
            print("\nVerifying special token IDs:")
            for token in fast_tokenizer.all_special_tokens:
                print(f"{token}: {fast_tokenizer.convert_tokens_to_ids(token)}")

            return fast_tokenizer

        if not model_config.pretrained:
            print("Training new tokenizer on dataset")
            return self._train_tokenizer(data_config.dataset)

        print(f"Using pretrained tokenizer from {model_config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _train_tokenizer(self, dataset_name: str) -> PreTrainedTokenizer:
        """Train a simple tokenizer on any text dataset."""
        print(f"Training new tokenizer for dataset: {dataset_name}")

        # Initialize with BPE model
        tokenizer = Tokenizer(models.BPE())

        # Basic normalizer
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Strip(),
                normalizers.Lowercase(),
                normalizers.NFD(),
                normalizers.StripAccents(),
            ]
        )

        # Enhanced pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.Punctuation(),
                pre_tokenizers.Whitespace(),
            ]
        )

        # Get dataset config and prepare training data
        dataset_config = DatasetConfigurator.get_dataset_config(dataset_name)
        print("Loading dataset for tokenizer training...")
        ds = load_dataset(dataset_config["name"], dataset_config.get("subset"))
        train_data = ds[dataset_config["train_field"]]

        def get_training_corpus():
            for i in range(0, len(train_data), 1000):
                batch = train_data[i : i + 1000]
                yield [str(text) for text in batch["text"] if text is not None]

        # Define special tokens with unique IDs
        special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[SEP]": 4,
        }

        # Train with more robust parameters and explicit special token IDs
        trainer = trainers.BpeTrainer(
            vocab_size=8000,
            special_tokens=list(special_tokens.keys()),
            initial_alphabet=[
                str(i) for i in range(10)
            ],  # Add digits to initial alphabet
            min_frequency=2,
            show_progress=True,
        )

        print("Training tokenizer...")
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

        # Explicitly set token IDs for special tokens
        for token, id_ in special_tokens.items():
            tokenizer.add_special_tokens([token])
            tokenizer.token_to_id(token)  # Ensure token is in vocabulary

        # Post-processor for adding BOS/EOS tokens
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [SEP] $B [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ],
        )

        # Save tokenizer
        save_path = self._get_tokenizer_path(dataset_name)
        print(f"Saving tokenizer files to {save_path}")

        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path / "tokenizer.json"))

        # Create and return the PreTrainedTokenizerFast
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
        )

        print("\nExample tokenizations:")
        test_strings = train_data["text"][:2]
        for test in test_strings:
            ids = fast_tokenizer.encode(str(test))
            decoded = fast_tokenizer.decode(ids)
            print(f"'{test}' -> {ids} -> '{decoded}'")

        # Verify special token IDs
        print("\nVerifying special token IDs:")
        for token in special_tokens:
            print(f"{token}: {fast_tokenizer.convert_tokens_to_ids(token)}")

        return fast_tokenizer
