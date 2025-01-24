import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict
import numpy as np
from tqdm import tqdm

from experiment.configs import DataConfig, ModelConfig, TrainingConfig
from experiment.models import DefaultLightningModule
from experiment.datasets.synthetic_datasets import (
    ArithmeticDataset,
    PatternDataset,
    ComplexArithmeticReasoningDataset,
)


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
        num_eval_samples: int = 10000,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_eval_samples = num_eval_samples

        self.datasets = {
            "arithmetic": lambda: ArithmeticDataset(max_len=50, min_len=3),
            "pattern": lambda: PatternDataset(seq_length=5),
            "complex_arithmetic_reasoning": lambda: ComplexArithmeticReasoningDataset(
                max_len=15, min_len=8
            ),
        }

        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataset_name: str) -> Dict[str, float]:
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Create dataset and sample evaluation examples
        dataset = self.datasets[dataset_name]()
        eval_samples = []

        for i, sample in enumerate(dataset):
            if i >= self.num_eval_samples:
                break
            eval_samples.append(sample)

        # Create dataloader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            eval_samples, batch_size=self.eval_batch_size, collate_fn=self._collate_fn
        )

        # Use appropriate evaluation method
        if dataset_name == "arithmetic":
            return self._evaluate_arithmetic_task(dataloader)
        elif dataset_name == "pattern":
            return self._evaluate_pattern_completion_task(dataloader)
        elif dataset_name == "complex_arithmetic_reasoning":
            return self._evaluate_complex_arithmetic_task(dataloader)

    def _collate_fn(self, batch):
        # Implement batching logic similar to your BatchCollator
        input_texts = [item["text"] for item in batch]
        encodings = self.tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt"
        )

        labels = encodings["input_ids"].clone()

        # Handle loss masks if present
        if "loss_mask" in batch[0]:
            loss_masks = [torch.tensor(item["loss_mask"]) for item in batch]
            max_len = max(mask.size(0) for mask in loss_masks)
            padded_masks = torch.zeros(len(loss_masks), max_len)
            for i, mask in enumerate(loss_masks):
                padded_masks[i, : len(mask)] = mask
            encodings["loss_mask"] = padded_masks

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

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
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output).split("[EOS]")[0]
                    true_text = self.tokenizer.decode(
                        batch["labels"][i], skip_special_tokens=True
                    )

                    try:
                        pred = float(pred_text.split("=")[-1].strip().replace(" ", ""))
                        target = float(
                            true_text.split("=")[-1].strip().replace(" ", "")
                        )

                        rel_error = abs(pred - target) / (abs(target) + 1e-8)
                        relative_errors.append(rel_error)

                        if rel_error < 0.01:
                            correct += 1
                    except Exception as e:
                        pass
                    total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "mean_relative_error": (
                np.mean(relative_errors) if relative_errors else float("inf")
            ),
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
                        pred = int(pred_text.split("->")[-1].strip())
                        target = int(true_text.split("->")[-1].strip())

                        if pred == target:
                            correct += 1

                        rel_error = abs(pred - target) / (abs(target) + 1e-8)
                        relative_errors.append(rel_error)
                    except:
                        pass
                    total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "mean_relative_error": (
                np.mean(relative_errors) if relative_errors else float("inf")
            ),
        }

    def _get_arithmetic_reasoning_input(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        batch_size = input_ids.size(0)

        # Batch decode
        seq_lengths = attention_mask.sum(dim=1)
        texts = self.tokenizer.batch_decode(
            [input_ids[i, : seq_lengths[i]] for i in range(batch_size)]
        )

        # Extract queries and append Answer:
        queries = [text.split("Answer:")[0].strip() + " Answer:" for text in texts]

        # Batch encode with padding
        encodings = self.tokenizer(
            queries, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(input_ids.device)

        return self._shift_padding_left(
            encodings["input_ids"], encodings["attention_mask"]
        )

    def _evaluate_complex_arithmetic_task(
        self, dataloader: DataLoader
    ) -> Dict[str, float]:
        correct = total = 0
        relative_errors = []
        step_accuracy = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids, attention_mask = self._get_arithmetic_reasoning_input(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )

                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=200,  # Increased for longer reasoning chains
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output).split("[EOS]")[0]
                    true_text = self.tokenizer.decode(
                        batch["labels"][i], skip_special_tokens=True
                    )

                    try:
                        # Extract final answer
                        pred_steps = pred_text.split("Answer:")[1].strip().split(")")
                        true_steps = true_text.split("Answer:")[1].strip().split(")")

                        # Get final values
                        pred_final = float(pred_steps[-2].split("=")[1].strip())
                        true_final = float(true_steps[-2].split("=")[1].strip())

                        # Calculate relative error for final answer
                        rel_error = abs(pred_final - true_final) / (
                            abs(true_final) + 1e-8
                        )
                        relative_errors.append(rel_error)

                        # Check step accuracy
                        correct_steps = 0
                        total_steps = min(len(pred_steps), len(true_steps))

                        for j in range(total_steps - 1):  # -1 to skip empty last split
                            try:
                                pred_val = float(pred_steps[j].split("=")[1].strip())
                                true_val = float(true_steps[j].split("=")[1].strip())
                                if (
                                    abs(pred_val - true_val) / (abs(true_val) + 1e-8)
                                    < 0.01
                                ):
                                    correct_steps += 1
                            except:
                                continue

                        step_accuracy.append(correct_steps / total_steps)

                        if rel_error < 0.01:
                            correct += 1
                    except Exception as e:
                        pass
                    total += 1

        return {
            "final_accuracy": correct / total if total > 0 else 0,
            "mean_relative_error": (
                np.mean(relative_errors) if relative_errors else float("inf")
            ),
            "step_accuracy": np.mean(step_accuracy) if step_accuracy else 0,
        }
