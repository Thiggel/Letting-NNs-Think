import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, Any
from datasets import load_dataset
import numpy as np

from experiment.models import DefaultLightningModule
from experiment.datasets import BatchCollator


class SyntheticDatasetEvaluator:
    """Evaluator for synthetic datasets with specialized metrics"""

    def __init__(
        self,
        model: DefaultLightningModule,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataset_name: str) -> Dict[str, float]:
        # Load dataset
        dataset = load_dataset(f"flaitenberger/synthetic_{dataset_name}", split="test")

        # Create dataloader
        collator = BatchCollator(self.tokenizer, max_length=1024)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=collator, num_workers=4
        )

        metrics = {
            "copy_task": self._evaluate_copy_task,
            "reverse_task": self._evaluate_reverse_task,
            "sort_task": self._evaluate_sort_task,
            "arithmetic_task": self._evaluate_arithmetic_task,
            "pattern_completion_task": self._evaluate_pattern_completion_task,
            "bracket_matching_task": self._evaluate_bracket_matching_task,
        }

        if dataset_name not in metrics:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return metrics[dataset_name](dataloader)

    def _evaluate_copy_task(self, dataloader: DataLoader) -> Dict[str, float]:
        correct = total = 0

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    max_new_tokens=100,
                    attention_mask=batch["attention_mask"].to(self.device),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    true_text = self.tokenizer.decode(
                        batch["labels"][i], skip_special_tokens=True
                    )

                    # Extract prediction and target from texts
                    pred = pred_text.split(" -> ")[-1].strip()
                    target = true_text.split(" -> ")[-1].strip()

                    if pred == target:
                        correct += 1
                    total += 1

        return {"accuracy": correct / total if total > 0 else 0}

    def _evaluate_sort_task(self, dataloader: DataLoader) -> Dict[str, float]:
        correct = total = 0

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    max_new_tokens=100,
                    attention_mask=batch["attention_mask"].to(self.device),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    true_text = self.tokenizer.decode(
                        batch["labels"][i], skip_special_tokens=True
                    )

                    # Extract and compare sorted sequences
                    pred = pred_text.split(" -> ")[-1].strip().split()
                    target = true_text.split(" -> ")[-1].strip().split()

                    if pred == target:
                        correct += 1
                    total += 1

        return {"accuracy": correct / total if total > 0 else 0}

    def _evaluate_arithmetic_task(self, dataloader: DataLoader) -> Dict[str, float]:
        correct = total = 0
        relative_errors = []

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    max_new_tokens=20,
                    attention_mask=batch["attention_mask"].to(self.device),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    true_text = self.tokenizer.decode(
                        batch["labels"][i], skip_special_tokens=True
                    )

                    try:
                        pred = float(pred_text.split("=")[-1].strip())
                        target = float(true_text.split("=")[-1].strip())

                        # Calculate relative error
                        rel_error = abs(pred - target) / (abs(target) + 1e-8)
                        relative_errors.append(rel_error)

                        # Consider correct if relative error is small
                        if rel_error < 0.01:  # 1% relative error threshold
                            correct += 1
                    except (ValueError, IndexError):
                        pass
                    total += 1

        mean_relative_error = (
            np.mean(relative_errors) if relative_errors else float("inf")
        )

        return {
            "accuracy": correct / total if total > 0 else 0,
            "mean_relative_error": mean_relative_error,
        }

    def _evaluate_pattern_completion_task(
        self, dataloader: DataLoader
    ) -> Dict[str, float]:
        correct = total = 0
        relative_errors = []

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    max_new_tokens=20,
                    attention_mask=batch["attention_mask"].to(self.device),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    true_text = self.tokenizer.decode(
                        batch["labels"][i], skip_special_tokens=True
                    )

                    try:
                        pred = int(pred_text.split(" -> ")[-1].strip())
                        target = int(true_text.split(" -> ")[-1].strip())

                        if pred == target:
                            correct += 1

                        rel_error = abs(pred - target) / (abs(target) + 1e-8)
                        relative_errors.append(rel_error)
                    except (ValueError, IndexError):
                        pass
                    total += 1

        mean_relative_error = (
            np.mean(relative_errors) if relative_errors else float("inf")
        )

        return {
            "accuracy": correct / total if total > 0 else 0,
            "mean_relative_error": mean_relative_error,
        }

    def _evaluate_bracket_matching_task(
        self, dataloader: DataLoader
    ) -> Dict[str, float]:
        correct = total = 0

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    max_new_tokens=10,
                    attention_mask=batch["attention_mask"].to(self.device),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    true_text = self.tokenizer.decode(
                        batch["labels"][i], skip_special_tokens=True
                    )

                    pred = pred_text.split(" -> ")[-1].strip()
                    target = true_text.split(" -> ")[-1].strip()

                    if pred == target:
                        correct += 1
                    total += 1

        return {"accuracy": correct / total if total > 0 else 0}

    def _evaluate_reverse_task(self, dataloader: DataLoader) -> Dict[str, float]:
        return self._evaluate_copy_task(
            dataloader
        )  # Same evaluation logic as copy task
