from typing import Any, List, Optional, Tuple, Union
import os
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from lightning import LightningDataModule
from datasets import Dataset, IterableDataset, load_dataset, disable_caching, config

from experiment.configs import DataConfig, ModelConfig, TrainingConfig

from .DatasetSplit import DatasetSplit
from .TokenizationProcessor import TokenizationProcessor
from .DatasetManager import DatasetManager
from .BatchCollator import BatchCollator
from .DatasetConfigurator import DatasetConfigurator


class LanguageDataModule(LightningDataModule):
    """Main data module for language modeling tasks"""

def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        tokenizer: PreTrainedTokenizer,
        seed: int,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        config.HF_DATASETS_TIMEOUT = 300
        self.data_config = data_config
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.seed = seed
        self.dataset_manager = DatasetManager(
            cache_dir if cache_dir is not None 
            else (os.environ.get("BASE_CACHE_DIR", ".") if torch.cuda.is_available() else ".")
        )
        self.token_processor = TokenizationProcessor(tokenizer)
        self.batch_collator = BatchCollator(tokenizer, data_config.seq_length)
        self.datasets: Optional[DatasetSplit] = None
        self.setup()

    def prepare_data(self) -> None:
        disable_caching()

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset_config = DatasetConfigurator.get_dataset_config(self.data_config.dataset)
        
        if self.dataset_config.get("process_on_the_fly", False):
            self.datasets = self._prepare_streaming_datasets()
        else:
            cache_path = self.dataset_manager.get_cache_path(
                self.data_config, self.model_config, self.seed
            )
            if self.dataset_manager.cached_datasets_exist(cache_path):
                self.datasets = self.dataset_manager.load_cached_datasets(cache_path)
            else:
                self.datasets = self._prepare_datasets()
                self.dataset_manager.save_datasets(self.datasets, cache_path)

    def _prepare_datasets(self) -> DatasetSplit:
        if self.dataset_config.get("streaming", False):
            return self._prepare_streaming_datasets()
        else:
            return self._prepare_static_datasets()

    def _prepare_static_datasets(self) -> DatasetSplit:
        ds = load_dataset(
            self.dataset_config["name"],
            self.dataset_config.get("subset"),
            trust_remote_code=True,
        )

        train_dataset = self._process_split(ds[self.dataset_config["train_field"]])

        if "test_field" in self.dataset_config:
            test_dataset = self._process_split(ds[self.dataset_config["test_field"]])
        elif "test_subset" in self.dataset_config:
            train_dataset, test_dataset = self._split_dataset(
                train_dataset, self.dataset_config["test_subset"]
            )
        else:
            test_dataset = None

        train_dataset, val_dataset = self._split_dataset(
            train_dataset,
            (
                int(len(train_dataset) * 0.1)
                if "val_subset" not in self.dataset_config
                else int(self.dataset_config["val_subset"])
            ),
        )

        return DatasetSplit(train_dataset, val_dataset, test_dataset)

    def _prepare_streaming_datasets(self) -> DatasetSplit:
        ds = load_dataset(
            self.dataset_config["name"],
            self.dataset_config.get("subset"),
            streaming=True,
            trust_remote_code=True,
        )

        process_fn = partial(
            self._process_streaming_sample,
            q_func=self.dataset_config["q_func"],
            ans_func=self.dataset_config["ans_func"],
        )

        train_dataset = ds[self.dataset_config["train_field"]].map(
            process_fn,
            remove_columns=ds[self.dataset_config["train_field"]].column_names
        )

        val_dataset = (
            ds[self.dataset_config["validation_field"]].map(
                process_fn,
                remove_columns=ds[self.dataset_config["validation_field"]].column_names
            )
            if "validation_field" in self.dataset_config
            else self._create_validation_set(train_dataset)
        )

        test_dataset = (
            ds[self.dataset_config["test_field"]].map(
                process_fn,
                remove_columns=ds[self.dataset_config["test_field"]].column_names
            )
            if "test_field" in self.dataset_config
            else None
        )

        return DatasetSplit(train_dataset, val_dataset, test_dataset)

    def _process_streaming_sample(
        self, sample: dict[str, Any], q_func: callable, ans_func: callable
    ) -> dict[str, Any]:
        full_text = q_func(sample) + ans_func(sample)
        
        tokenized = self.token_processor.tokenize_text(
            [full_text],
            max_length=self.data_config.seq_length if self.data_config.seq_length > 0 else None
        )

        if len(tokenized["input_ids"][0]) < self.data_config.seq_length:
            return None

        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"][0],
        }

    def _process_split(
        self, dataset: Union[Dataset, IterableDataset]
    ) -> Union[Dataset, IterableDataset]:
        return dataset.map(
            lambda samples: self._tokenize(samples),
            remove_columns=dataset.column_names,
            batched=True,
        )

    def _filter_max_len(
        self, tokenized: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply the length filtering
        filtered_input_ids = []
        filtered_attention_mask = []

        for input_ids, attention_mask in zip(
            tokenized["input_ids"], tokenized["attention_mask"]
        ):
            if len(input_ids) >= self.data_config.seq_length:
                filtered_input_ids.append(input_ids)
                filtered_attention_mask.append(attention_mask)

        return filtered_input_ids, filtered_attention_mask

    def _tokenize(self, samples: dict[str, List[Any]]) -> dict[str, List[Any]]:
        samples = [dict(zip(samples, i)) for i in zip(*samples.values())]
        full_text = [
            self.dataset_config["q_func"](sample)
            + self.dataset_config["ans_func"](sample)
            + (
                self.tokenizer.eos_token
                if not self.dataset_config.get("streaming", False)
                else ""
            )
            for sample in samples
        ]

        tokenized = self.token_processor.tokenize_text(
            full_text,
            max_length=(
                self.data_config.seq_length if self.data_config.seq_length > 0 else None
            ),
        )

        if self.dataset_config.get("streaming", False):
            input_ids, attention_mask = self._filter_max_len(tokenized)
        else:
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.copy(),
        }

    def _split_dataset(
        self, dataset: Dataset, split_size: Union[int, float]
    ) -> Tuple[Dataset, Dataset]:
        split = dataset.train_test_split(
            test_size=split_size, shuffle=True, seed=self.seed
        )
        return split["train"], split["test"]

    def _create_validation_set(self, train_dataset: IterableDataset) -> IterableDataset:
        validation_dataset = train_dataset.filter(
            lambda _, idx: idx < self.data_config.val_dataset_size, with_indices=True
        )
        return validation_dataset

    def train_dataloader(self) -> DataLoader:
        if not self.datasets or not self.datasets.train:
            raise ValueError("Training dataset not initialized")

        return DataLoader(
            self.datasets.train,
            batch_size=self.data_config.batch_size,
            shuffle=not isinstance(self.datasets.train, IterableDataset),
            collate_fn=self.batch_collator,
            num_workers=self._get_num_workers(),
            drop_last=not isinstance(self.datasets.train, IterableDataset),
        )

    def val_dataloader(self) -> DataLoader:
        if not self.datasets or not self.datasets.validation:
            raise ValueError("Validation dataset not initialized")

        return DataLoader(
            self.datasets.validation,
            batch_size=self.data_config.batch_size,
            collate_fn=self.batch_collator,
            num_workers=self._get_num_workers(),
            drop_last=not isinstance(self.datasets.validation, IterableDataset),
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.datasets or not self.datasets.test:
            return None

        return DataLoader(
            self.datasets.test,
            batch_size=self.data_config.batch_size,
            collate_fn=self.batch_collator,
            num_workers=self._get_num_workers(),
            drop_last=not isinstance(self.datasets.test, IterableDataset),
        )

    def _get_num_workers(self) -> int:
        if self.dataset_config.get("streaming", False):
            return 1

        if hasattr(os, "sched_getaffinity"):
            try:
                max_workers = len(os.sched_getaffinity(0))
            except Exception:
                max_workers = os.cpu_count() or 1
        else:
            max_workers = os.cpu_count() or 1

        return int(max_workers * 0.75)
