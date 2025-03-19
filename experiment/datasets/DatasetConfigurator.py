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
                "subset": "sample-10BT",
                "q_func": lambda x: x["text"],
                "ans_func": lambda x: "",
                "train_field": "train",
                "process_on_the_fly": True,
                "val_subset": 1000,
            },
            "cot_collection": {
                "name": "kaist-ai/CoT-Collection",
                "q_func": lambda x: f"<query>{x['source']}</query><thought>{x['rationale']}</thought>",
                "ans_func": lambda x: f"<answer>{x['target']}</answer>",
                "train_field": "train",
                "val_subset": 500,
                "process_on_the_fly": True,
                "synthetic": True,
            },
            "gsm8k": {
                "name": "gsm8k",
                "q_func": lambda x: f"Question: {x['question']}\n\nAnswer:",
                "ans_func": lambda x: " " + x["answer"],
                "subset": "main",
                "train_field": "train",
                "test_field": "test",
            },
            "csqa_gen": {
                "dataset_class": "CSQAGen",
                "q_func": lambda x: f"<query>{x['query']}</query>\n",
                "ans_func": lambda x: f"<steps>{x['steps']}</steps>\n<answer>{x['solution']}</answer>",
                "val_subset": 250,
                "synthetic": True,
                "process_on_the_fly": True,
            },
            "gsm8k_gen": {
                "dataset_class": "GSM8KGen",
                "q_func": lambda x: f"<query>{x['query']}</query>\n",
                "ans_func": lambda x: f"<steps>{x['steps']}</steps>\n<answer>{x['solution']}</answer>",
                "val_subset": 250,
                "synthetic": True,
                "process_on_the_fly": True,
            },
            "csqa_gsm8k_gen": {
                "dataset_class": "ReasoningDataset",
                "q_func": lambda x: f"<query>{x['query']}</query>\n",
                "ans_func": lambda x: f"<steps>{x['steps']}</steps>\n<answer>{x['solution']}</answer>",
                "val_subset": 250,
                "synthetic": True,
                "process_on_the_fly": True,
            },
            "arithmetic": {
                "dataset_class": "ArithmeticDataset",
                "q_func": lambda x: x["text"].split(" = ")[0],
                "ans_func": lambda x: " = " + x["text"].split(" = ")[1],
                "train_field": "train",
                "process_on_the_fly": True,
                "val_subset": 1000,
                "dataset_params": {"max_len": 50, "min_len": 3},
                "synthetic": True,
            },
            "complex_arithmetic_reasoning": {
                "dataset_class": "ComplexArithmeticReasoningDataset",
                "q_func": lambda x: x["text"].split("Answer:")[0],
                "ans_func": lambda x: "Answer:" + x["text"].split("Answer:")[1],
                "train_field": "train",
                "process_on_the_fly": True,
                "val_subset": 1000,
                "dataset_params": {"max_len": 30, "min_len": 15},
                "synthetic": True,
            },
            "pattern": {
                "dataset_class": "PatternDataset",
                "q_func": lambda x: x["text"].split(" -> ")[0],
                "ans_func": lambda x: " -> " + x["text"].split(" -> ")[1],
                "train_field": "train",
                "process_on_the_fly": True,
                "val_subset": 1000,
                "dataset_params": {"seq_length": 5},
                "synthetic": True,
            },
        }

        return base_configs
