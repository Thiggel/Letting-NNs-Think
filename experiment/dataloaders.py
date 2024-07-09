import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, disable_caching
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from experiment.utils.args import Args


def create_dataloaders(
    tokenizer: PreTrainedTokenizer, args: Args
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    disable_caching()

    cache_path: str = get_cache_path(args)

    if cached_datasets_exist(cache_path):
        return load_cached_dataloaders(cache_path, args)

    dataset_config: Dict[str, Any] = get_dataset_config(args.dataset)

    train_dataset, val_dataset, test_dataset = prepare_datasets(
        args, tokenizer, dataset_config
    )

    save_datasets_to_disk(train_dataset, val_dataset, test_dataset, cache_path)

    return create_and_return_dataloaders(train_dataset, val_dataset, test_dataset, args)


def get_cache_path(args: Args) -> str:
    return f"./cached_datasets/{args.model_name}_{args.dataset}_{args.seq_length}_{args.train_batch_size}_{args.seed}"


def cached_datasets_exist(cache_path: str) -> bool:
    return all(
        os.path.exists(f"{cache_path}_{split}") for split in ["train", "valid", "test"]
    )


def load_cached_dataloaders(
    cache_path: str, args: Args
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset: Dataset = load_from_disk(f"{cache_path}_train")
    val_dataset: Dataset = load_from_disk(f"{cache_path}_valid")
    test_dataset: Dataset = load_from_disk(f"{cache_path}_test")

    return create_and_return_dataloaders(train_dataset, val_dataset, test_dataset, args)


def get_all_dataset_configs() -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {
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
            "ans_func": lambda x: " " + (x["sol1"] if x["label"] == 0 else x["sol2"]),
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
        "ultrafeedback": {
            "name": "openbmb/UltraFeedback",
            "q_func": lambda x: f"Question: {x['instruction']}\n\nAnswer:",
            "ans_func": lambda x: " " + x["completions"][0]["response"],
            "train_field": "train",
            "test_subset": 1000,
            "custom_filter": lambda ds: ds.filter(
                lambda x: len([c for c in x["completions"] if c["model"] == "gpt-4"])
                > 0
            ),
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


def get_all_dataset_names() -> List[str]:
    return list(get_all_dataset_configs().keys())


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    configs: Dict[str, Dict[str, Any]] = get_all_dataset_configs()

    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return configs[dataset_name]


def prepare_datasets(
    args: Args, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    def tokenize(samples: Dict[str, List[Any]], args: Args) -> Dict[str, List[Any]]:
        samples = [dict(zip(samples, i)) for i in zip(*samples.values())]
        questions: List[str] = [config["q_func"](sample) for sample in samples]
        full: List[str] = [
            config["q_func"](sample) + config["ans_func"](sample) for sample in samples
        ]

        questions_tokenized: Dict[str, List[Any]] = tokenize_text(
            questions, tokenizer, args
        )
        full_labels: Dict[str, List[Any]] = tokenize_text(full, tokenizer, args)

        question_sizes: List[int] = [
            len([_q for _q in q if _q != tokenizer.pad_token_id])
            for q in questions_tokenized["input_ids"]
        ]

        return {
            "input_ids": full_labels["input_ids"],
            "attention_mask": full_labels["attention_mask"],
            "question_sizes": question_sizes,
        }

    ds: DatasetDict = load_dataset(config["name"], config.get("subset"))

    if "custom_filter" in config:
        ds = config["custom_filter"](ds)

    train_dataset: Dataset = process_split(
        ds[config["train_field"]], args, tokenizer, tokenize
    )

    if "test_field" in config:
        test_dataset: Dataset = process_split(
            ds[config["test_field"]], args, tokenizer, tokenize
        )
    elif "test_subset" in config:
        train_dataset, test_dataset = split_dataset(
            train_dataset, config["test_subset"]
        )
    else:
        test_dataset = None

    train_dataset, val_dataset = split_dataset(
        train_dataset, int(len(train_dataset) * 0.1)
    )

    return train_dataset, val_dataset, test_dataset


def tokenize_text(
    text: List[str], tokenizer: PreTrainedTokenizer, args: Args
) -> Dict[str, List[Any]]:
    return tokenizer(
        text,
        padding="max_length" if args.train_batch_size > 1 else "do_not_pad",
        truncation=args.seq_length > 0,
        max_length=args.seq_length if args.seq_length > 0 else None,
    )


def process_split(
    dataset: Dataset,
    args: Args,
    tokenizer: PreTrainedTokenizer,
    tokenize_func: Callable,
) -> Dataset:
    dataset = dataset.map(
        lambda samples: tokenize_func(samples, args),
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=24,
    )
    dataset = filter_dataset(dataset, tokenizer)
    return dataset.with_format("torch")


def filter_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    return dataset.filter(
        lambda samples: [
            ids[-1] == tokenizer.eos_token_id or ids[-1] == tokenizer.pad_token_id
            for ids in samples["input_ids"]
        ],
        batched=True,
        num_proc=24,
    )


def split_dataset(
    dataset: Dataset, split_size: Union[int, float]
) -> Tuple[Dataset, Dataset]:
    split = dataset.train_test_split(test_size=split_size, shuffle=True, seed=42)
    return split["train"], split["test"]


def save_datasets_to_disk(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset],
    cache_path: str,
) -> None:
    train_dataset.save_to_disk(f"{cache_path}_train")
    val_dataset.save_to_disk(f"{cache_path}_valid")
    if test_dataset:
        test_dataset.save_to_disk(f"{cache_path}_test")


def create_and_return_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset],
    args: Args,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size
    )
    test_dataloader: Optional[DataLoader] = (
        DataLoader(test_dataset, batch_size=args.eval_batch_size)
        if test_dataset
        else None
    )

    print(
        f"train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset) if test_dataset else 'N/A'}"
    )
    return train_dataloader, val_dataloader, test_dataloader
