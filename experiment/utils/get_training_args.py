import argparse
from typing import Union, Literal

from experiment.datasets import LanguageDataModule
from experiment.utils import Args


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


def parse_num_steps(value):
    try:
        # Try converting to an integer
        return int(value)
    except ValueError:
        # If not an int, check if it's one of the allowed string literals
        if value in {"classifier", "fixed_point", "random"}:
            return value
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid value: {value}. Must be an int or one of ['classifier', 'fixed_point']."
            )


def parse_index_or_interval(value):
    # it's either a number, or two numbers separated by a colon
    try:
        return int(value)
    except ValueError:
        try:
            return tuple(map(int, value.split(":")))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value: {value}. Must be an int or two ints separated by a colon."
            )


def parse_list(arg):
    return [item.strip() for item in arg.split(",")]


def get_training_args(get_defaults: bool = False) -> Args:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--seeds", type=int_list, default=[1, 2, 3], help="Random seeds"
    )
    parser.add_argument("--num_runs", type=int, default=3, help="The number of runs")
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="The model name to be used"
    )
    parser.add_argument(
        "--finetune_layers",
        type=int_list_or_all,
        help="The layers to fine-tune",
    )
    parser.add_argument(
        "--make_layers_recurrent",
        type=parse_index_or_interval,
        default=None,
        help="The layer to make recurrent",
    )
    parser.add_argument(
        "--recurrent_mode",
        type=str,
        choices=["mamba", "transformer"],
        default="transformer",
    )
    parser.add_argument(
        "--num_steps",
        type=parse_num_steps,
        help="Number of steps as an integer or 'classifier'/'fixed_point'/'random'.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=LanguageDataModule.get_all_dataset_names(),
        default=LanguageDataModule.get_all_dataset_names()[0],
        help="The dataset to use for training",
    )
    parser.add_argument(
        "--seq_length", type=int, default=128, help="The maximum sequence length"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="The training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="The evaluation batch size"
    )
    parser.add_argument(
        "--no_logger",
        action="store_false",
        dest="logger",
        help="Whether to use a logger",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default",
        help="The name of the experiment",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=5, help="The maximum number of epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="The number of warmup steps"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The path to the checkpoint to load",
    )
    parser.add_argument(
        "--save_to_checkpoint",
        type=str,
        default=None,
        help="The path to the checkpoint to save to",
    )
    parser.add_argument(
        "--no_evaluate",
        action="store_false",
        dest="evaluate",
        help="Whether to evaluate the model",
    )
    parser.set_defaults(logger=True, evaluate=True)
    parser.add_argument(
        "--use_time_embedding",
        action="store_true",
        help="Whether to use a time embedding in the model",
    )
    parser.add_argument(
        "--use_gating",
        action="store_true",
        help="Whether to use a gating in the model",
    )
    parser.add_argument(
        "--use_random_intermediate_supervision",
        action="store_true",
        help="Whether to use random intermediate supervision",
    )
    parser.set_defaults(logger=True, evaluate=True, time_embedding=False, gating=False)

    parser.add_argument(
        "--training_procedure",
        type=str,
        choices=["default", "reinforce", "uninterrupted", "make_uninterrupted"],
        help="The training procedure to use",
    )

    parser.add_argument(
        "--evaluation_metrics",
        type=parse_list,
        help="The evaluation metrics to use",
    )

    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor")

    parser.add_argument(
        "--temperature", type=float, default=1.0, help="The sampling temperature"
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="The maximum gradient norm for gradient clipping",
    )

    parser.add_argument(
        "--baseline_decay",
        type=float,
        default=0.99,
        help="The decay factor for the baseline",
    )

    if get_defaults:
        return parser.parse_args([])

    return parser.parse_args()
