from typing import Any


class DatasetConfigurator:
    """Manages dataset configurations"""

    @staticmethod
    def get_dataset_config(dataset_name: str) -> dict[str, Any]:
        configs = DatasetConfigurator.get_all_dataset_configs()
        if dataset_name not in configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return configs[dataset_name]

    @staticmethod
    def get_all_dataset_configs() -> dict[str, dict[str, Any]]:
        base_configs = {
            "ultrafeedback": {
                "name": "openbmb/UltraFeedback",
                "q_func": lambda x: f"Question: {x['instruction']}\n\nAnswer:",
                "ans_func": lambda x: " " + x["completions"][0]["response"],
                "train_field": "train",
                "test_subset": 1000,
                "custom_filter": lambda ds: ds.filter(
                    lambda x: len(
                        [c for c in x["completions"] if c["model"] == "gpt-4"]
                    )
                    > 0
                ),
                "streaming": False,
            },
            "fineweb": {
                "name": "HuggingFaceFW/fineweb",
                "subset": "CC-MAIN-2023-50",
                "q_func": lambda x: x["text"],
                "ans_func": lambda x: "",
                "train_field": "train",
                "process_on_the_fly": True,
                "val_subset": 1000,
            },
            "gsm8k": {
                "name": "gsm8k",
                "q_func": lambda x: f"Question: {x['question']}\n\nAnswer:",
                "ans_func": lambda x: " " + x["answer"],
                "subset": "main",
                "train_field": "train",
                "test_field": "test",
            },
            "synthetic_arithmetic_task": {
                "name": "flaitenberger/synthetic_arithmetic_task",
                "train_field": "train",
                "q_func": lambda x: x["text"].split(" = ")[0],
                "ans_func": lambda x: " = " + x["text"].split(" = ")[1],
                "use_loss_mask": True,
                "streaming": False,
                "test_subset": 1000,
                # Pass through the loss mask from the dataset
                "process_func": lambda x: {
                    "text": x["text"],
                    "loss_mask": x["loss_mask"],
                    "input_len": x["input_len"],
                },
            },
        }

        # Add synthetic dataset configurations
        synthetic_tasks = [
            "copy_task",
            "reverse_task",
            "sort_task",
            "pattern_completion_task",
            "bracket_matching_task",
        ]

        for task in synthetic_tasks:
            base_configs[f"synthetic_{task}"] = {
                "name": f"flaitenberger/synthetic_{task}",
                "train_field": "train",
                "q_func": lambda x: x["text"].split(" -> ")[0],
                "ans_func": lambda x: " -> " + x["text"].split(" -> ")[1],
                "use_loss_mask": True,
                "streaming": False,
                "test_subset": 1000,
                # Pass through the loss mask from the dataset
                "process_func": lambda x: {
                    "text": x["text"],
                    "loss_mask": x["loss_mask"],
                    "input_len": x["input_len"],
                },
            }

        return base_configs
