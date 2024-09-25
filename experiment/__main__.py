import os
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

from experiment.utils import print_mean_std
from experiment.utils import run_different_seeds
from experiment.utils import get_training_args


def main():
    load_dotenv()
    args = get_training_args()
    print(args)

    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    if args.logger:
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)

    all_results = run_different_seeds(args)

    print_mean_std(all_results)


if __name__ == "__main__":
    main()
