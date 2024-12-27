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
        num_eval_samples: int = 1000,  # New parameter for evaluation size
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_eval_samples = num_eval_samples

        # Create datasets directly instead of using DataModule
        from experiment.datasets import ArithmeticDataset, PatternDataset

        self.datasets = {
            "arithmetic": lambda: ArithmeticDataset(max_len=50, min_len=3),
            "pattern": lambda: PatternDataset(seq_length=5),
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

    def _evaluate_arithmetic_task(self, dataloader: DataLoader) -> Dict[str, float]:
        correct = total = 0
        relative_errors = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
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

                        rel_error = abs(pred - target) / (abs(target) + 1e-8)
                        relative_errors.append(rel_error)

                        if rel_error < 0.01:
                            correct += 1
                    except:
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
