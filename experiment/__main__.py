from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment.utils.make_layers_finetunable import make_layers_finetunable
from experiment.utils.remove_layers import remove_layers
from experiment.utils.add_pad_token import add_pad_token
from experiment.utils.set_seed import set_seed
from experiment.utils.get_training_args import get_training_args
from experiment.dataloaders import create_dataloaders


def main():
    args = get_training_args()
    print(args)

    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    add_pad_token(tokenizer)
    make_layers_finetunable(model, args.finetune_layers)
    remove_layers(model, args.remove_layers)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        tokenizer, args
    )

    print(model)
    print(tokenizer)


if __name__ == "__main__":
    main()


# TODO:
# 1. Lightning training loop
# 2. Add logging (wandb)
