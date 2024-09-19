import argparse
from typing import Union, Literal

from experiment.LanguageDataModule import LanguageDataModule
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


def get_training_args(get_defaults: bool = False) -> Args:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--seeds", type=int_list, default=[1, 2, 3], help="Random seeds"
    )
    parser.add_argument("--num_runs", type=int, default=1, help="The number of runs")
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="The model name to be used"
    )
    parser.add_argument(
        "--finetune_layers",
        type=int_list_or_all,
        help="The layers to fine-tune",
    )
    parser.add_argument(
        "--remove_layers", type=int_list, default=[], help="The layers to remove"
    )
    parser.add_argument(
        "--make_layer_recurrent",
        type=int,
        default=None,
        help="The layer to make recurrent",
    )
    parser.add_argument(
        "--recurrent_mode",
        type=str,
        choices=["ssm", "mamba", "adaptive_transformer"],
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=3,
        help="The number of steps in the recurrent transformer",
    )
    parser.add_argument(
        "--use_skip_connection",
        action="store_true",
        help="Whether to use a skip connection in the SSM",
    )
    parser.add_argument(
        "--use_fixed_num_steps",
        action="store_true",
        help="Whether to use a fixed number of steps in the recurrent transformer",
    )
    parser.add_argument(
        "--use_random_num_steps",
        action="store_true",
        help="Whether to use a random number of steps in the recurrent transformer",
    )
    parser.set_defaults(use_skip_connection=False, use_fixed_num_steps=False)
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
        "--max_epochs", type=int, default=10, help="The maximum number of epochs"
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
        "--no_evaluate",
        action="store_false",
        dest="evaluate",
        help="Whether to evaluate the model",
    )
    parser.set_defaults(logger=True, evaluate=True)
    parser.add_argument(
        "--time_embedding",
        action="store_true",
        help="Whether to use a time embedding in the model",
    )
    parser.add_argument(
        "--gating",
        action="store_true",
        help="Whether to use a gating in the model",
    )
    parser.set_defaults(logger=True, evaluate=True, time_embedding=False, gating=False)

    if get_defaults:
        return parser.parse_args([])

    return parser.parse_args()
