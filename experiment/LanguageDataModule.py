import os
from typing import Any, Dict, List, Optional, Tuple, Union
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, disable_caching
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from lightning import LightningDataModule
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from experiment.LMLightningModule import LMLightningModule
from experiment.utils.get_num_workers import get_num_workers
from experiment.utils.args import Args


class LanguageDataModule(LightningDataModule):
    def __init__(
        self,
        model: LMLightningModule,
        tokenizer: PreTrainedTokenizer,
        args: Args,
        seed: int,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.seed = seed
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.hidden_states_cache_file = self.get_hidden_states_cache_file()
        self.setup()

    def prepare_data(self):
        disable_caching()

    def setup(self, stage: Optional[str] = None):
        if os.path.exists(self.hidden_states_cache_file):
            self.load_hidden_states()
        else:
            cache_path: str = self.get_cache_path()

            if self.cached_datasets_exist(cache_path):
                self.load_cached_datasets(cache_path)
            else:
                dataset_config: Dict[str, Any] = self.get_dataset_config(
                    self.args.dataset
                )
                self.prepare_datasets(dataset_config)
                self.save_datasets_to_disk(cache_path)
            self.cache_hidden_states()

    def get_hidden_states_cache_file(self) -> str:
        last_frozen_layer_idx = self.model.get_idx_of_last_frozen_layer()
        cache_dir = os.environ["BASE_CACHE_DIR"] if torch.cuda.is_available() else "."
        cache_file = f"{cache_dir}/cached_hidden_states/{self.args.model_name}_{self.args.dataset}_{last_frozen_layer_idx}.pt"
        return cache_file

    def load_hidden_states(self):
        print(f"Loading hidden states from {self.hidden_states_cache_file}")
        hidden_states = torch.load(self.hidden_states_cache_file)
        self.train_dataset = hidden_states["train"]
        self.val_dataset = hidden_states["val"]
        if "test" in hidden_states:
            self.test_dataset = hidden_states["test"]

    def cache_hidden_states(self):
        print(f"Caching hidden states to {self.hidden_states_cache_file}")
        hidden_states = {
            "train": self.process_dataset_for_hidden_states(self.train_dataset),
            "val": self.process_dataset_for_hidden_states(self.val_dataset),
        }
        if self.test_dataset:
            hidden_states["test"] = self.process_dataset_for_hidden_states(
                self.test_dataset
            )
        torch.save(hidden_states, self.hidden_states_cache_file)

    def process_dataset_for_hidden_states(self, dataset: Dataset) -> Dataset:
        processed_hidden_states = []
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=int(0.75 * get_num_workers()),
        )

        last_frozen_layer_idx = self.model.get_idx_of_last_frozen_layer()

        self.model.eval()  # Ensure the model is in evaluation mode for generating hidden states
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing hidden states"):
                input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                last_hidden_state = outputs.hidden_states[last_frozen_layer_idx]
                processed_hidden_states.append(last_hidden_state.cpu())

        # Convert the list of hidden states back to a single torch tensor
        return torch.cat(processed_hidden_states, dim=0)

    def get_total_train_steps(self) -> int:
        return (
            len(self.train_dataset) // self.args.train_batch_size * self.args.max_epochs
        )

    def collate_fn(self, batch):
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )

        max_length = self.args.seq_length
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]
        labels = labels[:, :max_length]

        if input_ids.shape[1] < max_length:
            input_ids = F.pad(
                input_ids,
                (0, max_length - input_ids.shape[1]),
                value=self.tokenizer.pad_token_id,
            )
            attention_mask = F.pad(
                attention_mask, (0, max_length - attention_mask.shape[1]), value=0
            )
            labels = F.pad(labels, (0, max_length - labels.shape[1]), value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=int(0.75 * get_num_workers()),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=int(0.75 * get_num_workers()),
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.collate_fn,
                drop_last=True,
                num_workers=int(0.75 * get_num_workers()),
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
        ds: DatasetDict = load_dataset(config["name"], config.get("subset"))

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

    def process_split(self, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        dataset = dataset.map(
            lambda samples: self.tokenize(samples, config),
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=get_num_workers(),
        )
        return dataset.with_format("torch")

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

        # Shift the input_ids to create targets for next-token prediction
        input_ids = [ids[:-1] for ids in input_ids]
        labels = [ids[1:] for ids in input_ids]
        attention_mask = [mask[:-1] for mask in attention_mask]

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
            },
            "csqa_full": {
                "name": "tau/commonsense_qa",
                "q_func": lambda x: f"Question: {x['question']}\n\nChoices:\n{chr(10).join(x['choices']['text'])}\n\nAnswer:",
                "ans_func": lambda x: " "
                + x["choices"]["text"][x["choices"]["label"].index(x["answerKey"])],
                "train_field": "train",
                "test_field": "validation",
            },
            "arc_full": {
                "name": "allenai/ai2_arc",
                "q_func": lambda x: f"Question: {x['question']}\n\nChoices:\n{chr(10).join(x['choices']['text'])}\n\nAnswer:",
                "ans_func": lambda x: " "
                + x["choices"]["text"][x["choices"]["label"].index(x["answerKey"])],
                "train_field": "train",
                "test_field": "validation",
                "subset": "ARC-Challenge",
            },
            "piqa_full": {
                "name": "piqa",
                "q_func": lambda x: f"Question: {x['goal']}\n\nChoices:\n{x['sol1']}\n{x['sol2']}\n\nAnswer:",
                "ans_func": lambda x: " "
                + (x["sol1"] if x["label"] == 0 else x["sol2"]),
                "train_field": "train",
                "test_field": "validation",
            },
            "siqa_full": {
                "name": "social_i_qa",
                "q_func": lambda x: f"Question: Given the context, answer correctly the question.\nContext: {x['context']}\nQuestion: {x['question']}\n\nChoices:\n(0) {x['answerA']}\n(1) {x['answerB']}\n(2) {x['answerC']}\n\nAnswer:",
                "ans_func": lambda x: " " + f"({int(x['label']) - 1})",
                "train_field": "train",
                "test_field": "validation",
            },
            "openhermes": {
                "name": "teknium/openhermes",
                "q_func": lambda x: f"Question: {x['instruction']}{chr(10)}{x['input'] if x['input'] else ''}\n\nAnswer:",
                "ans_func": lambda x: " " + x["output"],
                "train_field": "train",
                "test_subset": 1000,
            },
            "alpaca": {
                "name": "yahma/alpaca-cleaned",
                "q_func": lambda x: f"Question: {x['instruction']}{chr(10)}{x['input'] if x['input'] else ''}\n\nAnswer:",
                "ans_func": lambda x: " " + x["output"],
                "train_field": "train",
                "test_subset": 1000,
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
