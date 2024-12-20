import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, Any
import numpy as np
from tqdm import tqdm

from experiment.configs import DataConfig, ModelConfig, TrainingConfig
from experiment.models import DefaultLightningModule
from experiment.datasets import LanguageDataModule


class SyntheticDatasetEvaluator:
    """Evaluator for synthetic datasets with specialized metrics"""

    def __init__(
        self,
        model: DefaultLightningModule,
        tokenizer: PreTrainedTokenizer,
        eval_batch_size: int = 32,
        data_config: DataConfig = None,
        model_config: ModelConfig = None,
        training_config: TrainingConfig = None,
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datamodule = LanguageDataModule(
            tokenizer=tokenizer,
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            eval_batch_size=eval_batch_size,
            seed=seed,
        )

        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataset_name: str) -> Dict[str, float]:
        dataloader = self.datamodule.test_dataloader()

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

    def _shift_padding_left(self, input_ids, attention_mask):
        """Shifts padding tokens from right to left side"""
        batch_size = input_ids.size(0)
        shifted_inputs = []
        shifted_masks = []

        for i in range(batch_size):
            # Get non-padded tokens
            seq_len = attention_mask[i].sum().item()
            valid_tokens = input_ids[i, :seq_len]

            # Create new tensor with left padding
            padded = torch.full_like(input_ids[i], self.tokenizer.pad_token_id)
            padded[-seq_len:] = valid_tokens

            # Create corresponding attention mask
            new_mask = torch.zeros_like(attention_mask[i])
            new_mask[-seq_len:] = 1

            shifted_inputs.append(padded)
            shifted_masks.append(new_mask)

        return torch.stack(shifted_inputs), torch.stack(shifted_masks)

    def _get_arithmetic_input(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        # Get the token index for the = symbol
        # Cut off the tokens after each = symbol in the batch (leave the = symbol)
        # Pad the sequence at the left side so that all = symbols are at the end and all sequences have the same length

        batch_size = input_ids.size(0)
        shifted_inputs = []
        shifted_masks = []

        equals_symbol = self.tokenizer.convert_tokens_to_ids("=")

        for i in range(batch_size):
            seq_len = attention_mask[i].sum().item()
            valid_tokens = input_ids[i, :seq_len]

            # Find the index of the = symbol
            equals_idx = (valid_tokens == equals_symbol).nonzero()

            if equals_idx.size(0) == 0:
                continue

            equals_idx = equals_idx[-1].item()
            valid_tokens = valid_tokens[: equals_idx + 1]

            # Create new tensor with left padding
            padded = torch.full_like(input_ids[i], self.tokenizer.pad_token_id)
            padded[-len(valid_tokens) :] = valid_tokens

            # Create corresponding attention mask
            new_mask = torch.zeros_like(attention_mask[i])
            new_mask[-len(valid_tokens) :] = 1

            shifted_inputs.append(padded)
            shifted_masks.append(new_mask)

        return torch.stack(shifted_inputs), torch.stack(shifted_masks)

    def _evaluate_arithmetic_task(self, dataloader: DataLoader) -> Dict[str, float]:
        correct = total = 0
        relative_errors = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids, attention_mask = self._get_arithmetic_input(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )

                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for i, output in tqdm(enumerate(outputs)):
                    pred_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    labels = batch["labels"][i]
                    labels = labels[labels != -100]
                    true_text = self.tokenizer.decode(labels, skip_special_tokens=True)

                    if i % 1000 == 0:
                        print(pred_text)
                        print(true_text)

                    try:
                        target = float(true_text.split("=")[-1].replace(" ", ""))
                        pred = float(pred_text.split("=")[-1].replace(" ", ""))
                        if i % 1000 == 0:
                            print(pred, target)
                            print()

                        # Calculate relative error
                        rel_error = abs(pred - target) / (abs(target) + 1e-8)
                        relative_errors.append(rel_error)

                        # Consider correct if relative error is small
                        if rel_error < 0.01:  # 1% relative error threshold
                            correct += 1
                    except Exception as e:
                        print(e)
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
                input_ids, attention_mask = self._shift_padding_left(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    attention_mask=attention_mask,
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
