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
        return {
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
            "c4": {
                "name": "allenai/c4",
                "subset_of_interest": "whole",
                "q_func": lambda x: x["text"],
                "ans_func": lambda x: "",
                "train_field": "train",
                "validation_field": "validation",  # Add this if The Pile has a validation split
                "streaming": True,
            },
            "gsm8k": {
                "name": "gsm8k",
                "q_func": lambda x: f"Question: {x['question']}\n\nAnswer:",
                "ans_func": lambda x: " " + x["answer"],
                "subset": "main",
                "train_field": "train",
                "test_field": "test",
            },
        }
