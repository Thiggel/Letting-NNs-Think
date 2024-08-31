from lm_eval import evaluator
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from lm_eval.models.huggingface import HFLM

from experiment.utils.args import Args


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    seed: int,
    args: Args,
    limit=None,
):
    wrapped_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        max_length=512,
        backend="causal",
    )

    return evaluator.simple_evaluate(
        model=wrapped_model,
        tasks=[
            "commonsense_qa",
            "gsm8k",
            # "mmlu",
            # "truthfulqa",
            "piqa",
        ],
        num_fewshot=0,
        batch_size=args.eval_batch_size,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        limit=limit,
        log_samples=True,
    )["results"]
