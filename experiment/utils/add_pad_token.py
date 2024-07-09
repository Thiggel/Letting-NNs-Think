from transformers import AutoTokenizer


def add_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
