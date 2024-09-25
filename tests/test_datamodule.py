import torch
from transformers import AutoTokenizer

from experiment.datasets import LanguageDataModule
from experiment.utils import get_training_args
from experiment.utils import add_pad_token


def test_datamodule():
    args = get_training_args(get_defaults=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    add_pad_token(tokenizer)

    datamodule = LanguageDataModule(tokenizer, args, 0)

    for batch in datamodule.train_dataloader():
        assert (
            batch["input_ids"].shape[0] == args.train_batch_size
        ), f"Expected {args.train_batch_size} but got {batch['input_ids'].shape[0]}"

        assert (
            batch["input_ids"].shape[1] == args.seq_length
        ), f"Expected {args.seq_length} but got {batch['input_ids'].shape[1]}"

        assert (
            batch["labels"].shape[0] == args.train_batch_size
        ), f"Expected {args.train_batch_size} but got {batch['labels'].shape[0]}"

        assert (
            batch["labels"].shape[1] == args.seq_length
        ), f"Expected {args.seq_length} but got {batch['labels'].shape[1]}"

        for index in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][index]
            labels = batch["labels"][index]

            # Assert that labels are shifted input_ids
            assert torch.equal(
                input_ids[labels != -100][1:], labels[labels != -100][:-1]
            ), f"Labels should be shifted input_ids for index {index}, but instead got {input_ids[labels != -100]} and {labels[labels != -100]}"

            # Assert that all -100 in labels correspond to padding tokens in input_ids
            pad_token_id = tokenizer.pad_token_id
            assert torch.all(
                (labels == -100) == (input_ids == pad_token_id)
            ), f"Mismatch between -100 in labels and pad tokens in input_ids for index {index}"

        break
