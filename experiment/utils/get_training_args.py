import argparse
from typing import Union, Literal

from experiment.dataloaders import get_all_dataset_names
from experiment.utils.args import Args


def int_list_or_all(value) -> Union[Literal["all"], list[int]]:
    if value == "all":
        return value
    try:
        # Split the string by commas and convert each part to an integer
        return [int(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Argument must be 'all' or a comma-separated list of integers."
        )


def int_list(value) -> list[int]:
    try:
        # Split the string by commas and convert each part to an integer
        return [int(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Argument must be a comma-separated list of integers."
        )


def get_training_args() -> Args:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="The model name to be used"
    )
    parser.add_argument(
        "--finetune_layers",
        type=int_list_or_all,
        default="all",
        help="The layers to fine-tune",
    )
    parser.add_argument(
        "--remove_layers", type=int_list, default=[], help="The layers to remove"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=get_all_dataset_names(),
        default=get_all_dataset_names()[0],
        help="The dataset to use for training",
    )
    parser.add_argument(
        "--seq_length", type=int, default=128, help="The maximum sequence length"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="The training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="The evaluation batch size"
    )
    return parser.parse_args()
