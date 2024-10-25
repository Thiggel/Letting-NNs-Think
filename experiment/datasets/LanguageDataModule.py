from typing import Any, List, Optional, Tuple, Union
import os
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from lightning import LightningDataModule
from datasets import (
    Dataset,
    IterableDataset,
    load_dataset,
    disable_caching,
)

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

        self.data_config = data_config
        self.model_config = model_config
        self.training_config = training_config

        self.tokenizer = tokenizer
        self.seed = seed

        self.dataset_manager = DatasetManager(
            cache_dir
            or (
                os.environ.get("BASE_CACHE_DIR", ".")
                if torch.cuda.is_available()
                else "."
            )
        )
        self.token_processor = TokenizationProcessor(tokenizer)
        self.batch_collator = BatchCollator(tokenizer, data_config.seq_length)
        self.datasets: Optional[DatasetSplit] = None

        self.setup()

    def prepare_data(self) -> None:
        disable_caching()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_config = DatasetConfigurator.get_dataset_config(
            self.data_config.dataset
        )
        cache_path = self.dataset_manager.get_cache_path(
            self.data_config, self.model_config, self.seed
        )

        if self.dataset_manager.cached_datasets_exist(cache_path):
            self.datasets = self.dataset_manager.load_cached_datasets(cache_path)
        else:
            self.datasets = self._prepare_datasets(dataset_config)
            self.dataset_manager.save_datasets(self.datasets, cache_path)

    def _prepare_datasets(self, dataset_config: dict[str, Any]) -> DatasetSplit:
        if dataset_config.get("streaming", False):
            return self._prepare_streaming_datasets(dataset_config)
        else:
            return self._prepare_static_datasets(dataset_config)

    def _prepare_static_datasets(self, dataset_config: dict[str, Any]) -> DatasetSplit:
        ds = load_dataset(
            dataset_config["name"], dataset_config.get("subset"), trust_remote_code=True
        )

        train_dataset = self._process_split(
            ds[dataset_config["train_field"]], dataset_config
        )

        if "test_field" in dataset_config:
            test_dataset = self._process_split(
                ds[dataset_config["test_field"]], dataset_config
            )
        elif "test_subset" in dataset_config:
            train_dataset, test_dataset = self._split_dataset(
                train_dataset, dataset_config["test_subset"]
            )
        else:
            test_dataset = None

        train_dataset, val_dataset = self._split_dataset(
            train_dataset, int(len(train_dataset) * 0.1)
        )

        return DatasetSplit(train_dataset, val_dataset, test_dataset)

    def _prepare_streaming_datasets(
        self, dataset_config: dict[str, Any]
    ) -> DatasetSplit:
        ds = load_dataset(
            dataset_config["name"],
            "en",
            streaming=True,
            trust_remote_code=True,
        )

        train_dataset = self._process_split(
            ds[dataset_config["train_field"]], dataset_config
        )
        val_dataset = (
            self._process_split(ds[dataset_config["validation_field"]], dataset_config)
            if "validation_field" in dataset_config
            else self._create_validation_set(train_dataset)
        )
        test_dataset = (
            self._process_split(ds[dataset_config["test_field"]], dataset_config)
            if "test_field" in dataset_config
            else None
        )

        return DatasetSplit(train_dataset, val_dataset, test_dataset)

    def _process_split(
        self, dataset: Union[Dataset, IterableDataset], dataset_config: dict[str, Any]
    ) -> Union[Dataset, IterableDataset]:
        return dataset.map(
            lambda samples: self._tokenize(samples, dataset_config),
            remove_columns=dataset.column_names,
            batched=True,
        )

    def _tokenize(
        self, samples: dict[str, List[Any]], dataset_config: dict[str, Any]
    ) -> dict[str, List[Any]]:
        samples = [dict(zip(samples, i)) for i in zip(*samples.values())]
        full_text = [
            dataset_config["q_func"](sample)
            + dataset_config["ans_func"](sample)
            + self.tokenizer.eos_token
            for sample in samples
        ]

        tokenized = self.token_processor.tokenize_text(
            full_text,
            max_length=(
                self.data_config.seq_length if self.data_config.seq_length > 0 else None
            ),
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }

    def _split_dataset(
        self, dataset: Dataset, split_size: Union[int, float]
    ) -> Tuple[Dataset, Dataset]:
        split = dataset.train_test_split(test_size=split_size, shuffle=True, seed=42)
        return split["train"], split["test"]

    def _create_validation_set(self, train_dataset: IterableDataset) -> IterableDataset:
        def validation_generator():
            train_iter = iter(train_dataset)
            for i, item in enumerate(train_iter):
                if i % 10 == 0:  # Every 10th item goes to validation
                    yield item
                else:
                    yield next(train_iter)

        return IterableDataset.from_generator(validation_generator)

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
        if hasattr(os, "sched_getaffinity"):
            try:
                max_workers = len(os.sched_getaffinity(0))
            except Exception:
                max_workers = os.cpu_count() or 1
        else:
            max_workers = os.cpu_count() or 1

        return int(max_workers * 0.75)
