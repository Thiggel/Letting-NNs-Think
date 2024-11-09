from pathlib import Path
import json
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedModel, PreTrainedTokenizer
import os


class ModelEvaluator:
    """Handles model evaluation using lm-eval-harness with multi-GPU support"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Initialize distributed setup if not already done
        if not dist.is_initialized():
            if "RANK" not in os.environ:
                os.environ["RANK"] = "0"
            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = "0"
            if "WORLD_SIZE" not in os.environ:
                os.environ["WORLD_SIZE"] = "1"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"

            dist.init_process_group(backend="nccl")

        # Get local rank for device assignment
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        # Move model to GPU and wrap in DDP
        self.model = self.model.to(self.device)
        if dist.get_world_size() > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def evaluate(
        self, metrics: list[str], seed: int, experiment_name: str
    ) -> dict[str, float]:
        # Set up wrapped model for evaluation
        wrapped_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=128,
            max_length=512,
            backend="causal",
            device=self.device,  # Use assigned device
        )

        # Run evaluation
        output = evaluator.simple_evaluate(
            model=wrapped_model,
            tasks=metrics or ["commonsense_qa", "gsm8k", "piqa"],
            num_fewshot=0,
            batch_size=128,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            device=self.device,  # Use assigned device
            log_samples=True,
        )

        # Only save results on main process
        if dist.get_rank() == 0:
            self._save_samples(output["samples"], seed, experiment_name)
            self._save_results(output["results"], experiment_name)

        # Make sure all processes are synced
        dist.barrier()

        return output["results"]

    def _save_results(self, results: dict, experiment_name: str):
        output_dir = Path(os.getenv("BASE_CACHE_DIR") or "") / "evaluation_results"
        output_dir.mkdir(exist_ok=True)

        results_path = output_dir / f"{experiment_name}.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

    def _save_samples(self, samples: dict, seed, experiment_name):
        try:
            sample_dir = Path(os.environ["BASE_CACHE_DIR"] or "") / "samples"
            sample_dir.mkdir(exist_ok=True)

            sample_path = sample_dir / f"{experiment_name}_{seed}.json"
            sample_path.write_text(json.dumps(samples))
        except Exception as e:
            print(f"Failed to save samples: {e}")

    def __del__(self):
        # Cleanup distributed process group
        if dist.is_initialized():
            dist.destroy_process_group()
