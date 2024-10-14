import os
from typing import Any, Dict, List, Optional, Tuple, Union
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk,
    disable_caching,
    IterableDataset,
)
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from lightning import LightningDataModule
import torch.nn.functional as F

from experiment.utils import get_num_workers
from experiment.utils import Args


class LanguageDataModule(LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: Args, seed: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.seed = seed
        self.train_dataset: Optional[Union[Dataset, IterableDataset]] = None
        self.val_dataset: Optional[Union[Dataset, IterableDataset]] = None
        self.test_dataset: Optional[Union[Dataset, IterableDataset]] = None

        self.setup()

    def prepare_data(self):
        disable_caching()

    def setup(self, stage: Optional[str] = None):
        self.dataset_config: Dict[str, Any] = self.get_dataset_config(self.args.dataset)

        if self.dataset_config.get("streaming", False):
            self.prepare_streaming_datasets(self.dataset_config)
        else:
            cache_path: str = self.get_cache_path()
            if self.cached_datasets_exist(cache_path):
                self.load_cached_datasets(cache_path)
            else:
                self.prepare_datasets(self.dataset_config)
                self.save_datasets_to_disk(cache_path)

    def get_total_train_steps(self) -> int:
        return (
            len(self.train_dataset) // self.args.train_batch_size * self.args.max_epochs
        )

    def ensure_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            return torch.tensor(x)

    def collate_fn(self, batch):
        input_ids = pad_sequence(
            [self.ensure_tensor(item["input_ids"]) for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [self.ensure_tensor(item["attention_mask"]) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [self.ensure_tensor(item["labels"]) for item in batch],
            batch_first=True,
            padding_value=-100,
        )

        max_length = self.args.seq_length
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]
        labels = labels[:, :max_length]

        if input_ids.shape[1] < max_length:
            pad_length = max_length - input_ids.shape[1]
            input_ids = F.pad(
                input_ids, (0, pad_length), value=self.tokenizer.pad_token_id
            )
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            labels = F.pad(labels, (0, pad_length), value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_num_workers(self) -> int:
        return int(0.75 * get_num_workers())

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.get_num_workers(),
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                drop_last=not self.dataset_config.get("streaming", False),
                num_workers=self.get_num_workers(),
            )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.val_dataset, IterableDataset):
            return DataLoader(
                self.val_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.get_num_workers(),
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.collate_fn,
                drop_last=not self.dataset_config.get("streaming", False),
                num_workers=self.get_num_workers(),
            )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset:
            if isinstance(self.test_dataset, IterableDataset):
                return DataLoader(
                    self.test_dataset,
                    batch_size=self.args.eval_batch_size,
                    collate_fn=self.collate_fn,
                    num_workers=self.get_num_workers(),
                )
            else:
                return DataLoader(
                    self.test_dataset,
                    batch_size=self.args.eval_batch_size,
                    collate_fn=self.collate_fn,
                    drop_last=not self.dataset_config.get("streaming", False),
                    num_workers=self.get_num_workers(),
                )
        return None

    def get_cache_path(self) -> str:
        cache_dir = os.environ["BASE_CACHE_DIR"] if torch.cuda.is_available() else "."
        return f"{cache_dir}/cached_datasets/{self.args.model_name}_{self.args.dataset}_{self.args.seq_length}_{self.args.train_batch_size}_{self.seed}"

    def cached_datasets_exist(self, cache_path: str) -> bool:
        return all(
            os.path.exists(f"{cache_path}_{split}")
            for split in ["train", "valid", "test"]
        )

    def load_cached_datasets(self, cache_path: str):
        self.train_dataset = load_from_disk(f"{cache_path}_train")
        self.val_dataset = load_from_disk(f"{cache_path}_valid")
        self.test_dataset = load_from_disk(f"{cache_path}_test")

    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        configs: Dict[str, Dict[str, Any]] = self.get_all_dataset_configs()
        if dataset_name not in configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return configs[dataset_name]

    def prepare_datasets(self, config: Dict[str, Any]):
        ds: DatasetDict = load_dataset(
            config["name"], config.get("subset"), trust_remote_code=True
        )

        if "custom_filter" in config:
            ds = config["custom_filter"](ds)

        self.train_dataset = self.process_split(ds[config["train_field"]], config)

        if "test_field" in config:
            self.test_dataset = self.process_split(ds[config["test_field"]], config)
        elif "test_subset" in config:
            self.train_dataset, self.test_dataset = self.split_dataset(
                self.train_dataset, config["test_subset"]
            )
        else:
            self.test_dataset = None

        self.train_dataset, self.val_dataset = self.split_dataset(
            self.train_dataset, int(len(self.train_dataset) * 0.1)
        )

    def prepare_streaming_datasets(self, config: Dict[str, Any]):
        ds = load_dataset(
            config["name"],
            "en",
            streaming=True,
            trust_remote_code=True,
        )

        self.train_dataset = self.process_split(ds[config["train_field"]], config)

        # Create a separate validation dataset
        if "validation_field" in config:
            self.val_dataset = self.process_split(
                ds[config["validation_field"]], config
            )
        else:
            # If no separate validation field, create a validation set from the training data
            self.val_dataset = self.create_validation_set(self.train_dataset, config)

        if "test_field" in config:
            self.test_dataset = self.process_split(ds[config["test_field"]], config)
        else:
            self.test_dataset = None

    def create_validation_set(
        self, train_dataset: IterableDataset, config: Dict[str, Any]
    ) -> IterableDataset:
        def validation_generator():
            train_iter = iter(train_dataset)
            for i, item in enumerate(train_iter):
                if i % 10 == 0:  # Every 10th item goes to validation
                    yield item
                else:
                    yield next(train_iter)

        return IterableDataset.from_generator(validation_generator)

    def process_split(
        self, dataset: Union[Dataset, IterableDataset], config: Dict[str, Any]
    ) -> Union[Dataset, IterableDataset]:
        return dataset.map(
            lambda samples: self.tokenize(samples, config),
            remove_columns=dataset.column_names,
            batched=True,
        )

    def tokenize(
        self, samples: Dict[str, List[Any]], config: Dict[str, Any]
    ) -> Dict[str, List[Any]]:
        samples = [dict(zip(samples, i)) for i in zip(*samples.values())]
        full_text: List[str] = [
            config["q_func"](sample) + config["ans_func"](sample) for sample in samples
        ]

        tokenized = self.tokenize_text(full_text)

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def tokenize_text(self, text: List[str]) -> Dict[str, List[Any]]:
        return self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=self.args.seq_length > 0,
            max_length=self.args.seq_length if self.args.seq_length > 0 else None,
        )

    def split_dataset(
        self, dataset: Dataset, split_size: Union[int, float]
    ) -> Tuple[Dataset, Dataset]:
        split = dataset.train_test_split(test_size=split_size, shuffle=True, seed=42)
        return split["train"], split["test"]

    def save_datasets_to_disk(self, cache_path: str):
        self.train_dataset.save_to_disk(f"{cache_path}_train")
        self.val_dataset.save_to_disk(f"{cache_path}_valid")
        if self.test_dataset:
            self.test_dataset.save_to_disk(f"{cache_path}_test")

    @staticmethod
    def get_all_dataset_configs() -> Dict[str, Dict[str, Any]]:
        configs: Dict[str, Dict[str, Any]] = {
            "ultrafeedback": {
                "name": "openbmb/UltraFeedback",
                "q_func": lambda x: f"Question: {x['instruction']}\n\nAnswer:",
                "ans_func": lambda x: " " + x["completions"][0]["response"],
                "train_field": "train",
                "test_subset": 1000,
                "custom_filter": lambda ds: ds.filter(
                    lambda x: len(
                        [c for c in x["completions"] if c["model"] == "gpt-4"]
                    )
                    > 0
                ),
                "streaming": False,
            },
            "c4": {
                "name": "allenai/c4",
                "subset_of_interest": "whole",
                "q_func": lambda x: x["text"],
                "ans_func": lambda x: "",
                "train_field": "train",
                "validation_field": "validation",  # Add this if The Pile has a validation split
                "streaming": True,
            },
            "gsm8k": {
                "name": "gsm8k",
                "q_func": lambda x: f"Question: {x['question']}\n\nAnswer:",
                "ans_func": lambda x: " " + x["answer"],
                "subset": "main",
                "train_field": "train",
                "test_field": "test",
            },
        }

        return configs

    @staticmethod
    def get_all_dataset_names() -> List[str]:
        return list(LanguageDataModule.get_all_dataset_configs().keys())
