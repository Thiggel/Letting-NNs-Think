from lm_eval import tasks, evaluator

from experiment.eval.ModelWrapper import ModelWrapper
from experiment.utils.args import Args

def evaluate(wrapped_model: ModelWrapper, seed: int, args: Args, limit=None):
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
        limit=limit
    )["results"]
