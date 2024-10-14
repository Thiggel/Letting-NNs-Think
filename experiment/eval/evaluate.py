from lm_eval import evaluator
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from lm_eval.models.huggingface import HFLM
import os
import json

from experiment.utils import Args


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    seed: int,
    args: Args,
    limit=None,
    filename_suffix="",
):
    wrapped_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=64,
        max_length=512,
        backend="causal",
    )

    evaluation_metrics = (
        args.evaluation_metrics
        if args.evaluation_metrics
        else [
            "commonsense_qa",
            "gsm8k",
            # "mmlu",
            # "truthfulqa",
            "piqa",
        ]
    )

    output = evaluator.simple_evaluate(
        model=wrapped_model,
        tasks=evaluation_metrics,
        num_fewshot=0,
        batch_size=64,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        limit=limit,
        log_samples=True,
    )
    try:
        sample_dir = os.environ["BASE_CACHE_DIR"] + "/samples"
        os.makedirs(sample_dir, exist_ok=True)
        with open(
            f"{sample_dir}/{args.experiment_name}_{seed}{filename_suffix}.json", "w"
        ) as f:
            json.dump(output["samples"], f)
    except Exception as e:
        print(e)

    return output["results"]
