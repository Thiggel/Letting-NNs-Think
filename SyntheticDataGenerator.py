import numpy as np
from dotenv import load_dotenv
import os
import pandas as pd
from typing import Tuple, Dict
from datasets import Dataset
import random
from tqdm import tqdm
from huggingface_hub import login


class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        np.random.seed(seed)
        random.seed(seed)
        self.separator = " -> "  # Separator between input and target

    def generate_copy_task(
        self,
        num_samples: int,
        min_len: int = 3,
        max_len: int = 50,
        num_distractors: int = 3,
    ) -> Dataset:
        """Generate sequences with distractors and their copies."""
        data = []
        for _ in tqdm(range(num_samples), desc="Generating copy task"):
            # Generate main sequence
            length = random.randint(min_len, max_len)
            main_sequence = list(map(str, np.random.randint(0, 100, size=length)))

            # Generate distractor sequences
            distractors = []
            for _ in range(num_distractors):
                dist_len = random.randint(min_len, max_len)
                distractor = list(map(str, np.random.randint(0, 100, size=dist_len)))
                distractors.append(" ".join(distractor))

            # Combine sequences
            input_seq = " | ".join(distractors + [" ".join(main_sequence)])
            target_seq = " ".join(main_sequence)

            # Calculate loss mask (0 for input, 1 for target)
            full_sequence = input_seq + self.separator + target_seq
            loss_mask = [0] * (len(input_seq) + len(self.separator)) + [1] * len(
                target_seq
            )

            data.append(
                {
                    "text": full_sequence,
                    "loss_mask": loss_mask,
                    "input_len": len(input_seq) + len(self.separator),
                }
            )
        return Dataset.from_pandas(pd.DataFrame(data))

    def generate_reverse_task(
        self, num_samples: int, min_len: int = 3, max_len: int = 50
    ) -> Dataset:
        """Generate sequences and their reversals with loss masking."""
        data = []
        for _ in tqdm(range(num_samples), desc="Generating reverse task"):
            length = random.randint(min_len, max_len)
            sequence = list(map(str, np.random.randint(0, 100, size=length)))
            input_seq = " ".join(sequence)
            target_seq = " ".join(sequence[::-1])

            full_sequence = input_seq + self.separator + target_seq
            loss_mask = [0] * (len(input_seq) + len(self.separator)) + [1] * len(
                target_seq
            )

            data.append(
                {
                    "text": full_sequence,
                    "loss_mask": loss_mask,
                    "input_len": len(input_seq) + len(self.separator),
                }
            )
        return Dataset.from_pandas(pd.DataFrame(data))

    def generate_sort_task(
        self, num_samples: int, min_len: int = 3, max_len: int = 50
    ) -> Dataset:
        """Generate sequences and their sorted versions with loss masking."""
        data = []
        for _ in tqdm(range(num_samples), desc="Generating sort task"):
            length = random.randint(min_len, max_len)
            sequence = list(map(str, np.random.randint(0, 1000, size=length)))
            input_seq = " ".join(sequence)
            target_seq = " ".join(sorted(sequence, key=int))

            full_sequence = input_seq + self.separator + target_seq
            loss_mask = [0] * (len(input_seq) + len(self.separator)) + [1] * len(
                target_seq
            )

            data.append(
                {
                    "text": full_sequence,
                    "loss_mask": loss_mask,
                    "input_len": len(input_seq) + len(self.separator),
                }
            )
        return Dataset.from_pandas(pd.DataFrame(data))

    def _generate_arithmetic_expression(self, num_steps: int) -> Tuple[str, float]:
        """Generate a complex arithmetic expression with given number of operations."""
        operators = ["+", "-", "*", "/"]
        numbers = []
        ops = []

        # Generate first number
        numbers.append(random.randint(1, 20))

        # Generate operations and numbers
        for _ in range(num_steps):
            op = random.choice(operators)
            # For division, ensure clean division when possible
            if op == "/":
                current_result = eval(
                    " ".join(
                        map(
                            str,
                            [numbers[0]]
                            + [f"{ops[i]} {numbers[i+1]}" for i in range(len(ops))],
                        )
                    )
                )
                num = random.randint(1, 10)
                while current_result % num != 0 and len(str(current_result / num)) > 5:
                    num = random.randint(1, 10)
            else:
                num = random.randint(1, 20)

            numbers.append(num)
            ops.append(op)

        # Construct expression
        expression_parts = []
        for i in range(len(numbers)):
            if i > 0:
                expression_parts.append(ops[i - 1])
            expression_parts.append(str(numbers[i]))

        expression = " ".join(expression_parts)
        result = eval(expression)

        # Round result to 2 decimal places if it's a float
        if isinstance(result, float):
            result = round(result, 2)

        return expression, result

    def generate_arithmetic_task(
        self, num_samples: int, min_steps: int = 5, max_steps: int = 20
    ) -> Dataset:
        """Generate complex arithmetic problems with multiple operations."""
        data = []
        for _ in tqdm(range(num_samples), desc="Generating arithmetic task"):
            num_steps = random.randint(min_steps, max_steps)
            expression, result = self._generate_arithmetic_expression(num_steps)

            full_sequence = f"{expression} = {result}"
            loss_mask = [0] * (len(expression) + 3) + [1] * len(
                str(result)
            )  # +3 for " = "

            data.append(
                {
                    "text": full_sequence,
                    "loss_mask": loss_mask,
                    "input_len": len(expression) + 3,
                }
            )
        return Dataset.from_pandas(pd.DataFrame(data))

    def generate_pattern_completion_task(
        self, num_samples: int, sequence_length: int = 5
    ) -> Dataset:
        """Generate arithmetic sequence pattern completion problems with loss masking."""
        data = []
        for _ in tqdm(range(num_samples), desc="Generating pattern completion task"):
            start = random.randint(-100, 100)
            step = random.randint(-20, 20)
            if step == 0:
                step = 1

            sequence = [start + i * step for i in range(sequence_length)]
            input_seq = " ".join(map(str, sequence[:-1]))
            target = str(sequence[-1])

            full_sequence = input_seq + self.separator + target
            loss_mask = [0] * (len(input_seq) + len(self.separator)) + [1] * len(target)

            data.append(
                {
                    "text": full_sequence,
                    "loss_mask": loss_mask,
                    "input_len": len(input_seq) + len(self.separator),
                }
            )
        return Dataset.from_pandas(pd.DataFrame(data))

    def generate_bracket_matching_task(
        self, num_samples: int, max_depth: int = 5
    ) -> Dataset:
        """Generate nested bracket sequences with validity classification and loss masking."""

        def generate_valid_brackets(depth: int) -> str:
            if depth <= 0:
                return ""
            if depth == 1:
                return "()"

            choices = [
                f"({generate_valid_brackets(depth-1)})",
                f"{generate_valid_brackets(depth-1)}{generate_valid_brackets(depth-1)}",
            ]
            return random.choice(choices)

        def generate_invalid_brackets(depth: int) -> str:
            valid = generate_valid_brackets(depth)
            mutation = random.choice(["add", "remove", "change"])
            if mutation == "add":
                pos = random.randint(0, len(valid))
                bracket = random.choice(["(", ")"])
                return valid[:pos] + bracket + valid[pos:]
            elif mutation == "remove":
                pos = random.randint(0, len(valid) - 1)
                return valid[:pos] + valid[pos + 1 :]
            else:  # change
                pos = random.randint(0, len(valid) - 1)
                new_bracket = ")" if valid[pos] == "(" else "("
                return valid[:pos] + new_bracket + valid[pos + 1 :]

        data = []
        for _ in tqdm(range(num_samples), desc="Generating bracket matching task"):
            depth = random.randint(1, max_depth)
            is_valid = random.choice([True, False])

            if is_valid:
                sequence = generate_valid_brackets(depth)
                target = "valid"
            else:
                sequence = generate_invalid_brackets(depth)
                target = "invalid"

            full_sequence = sequence + self.separator + target
            loss_mask = [0] * (len(sequence) + len(self.separator)) + [1] * len(target)

            data.append(
                {
                    "text": full_sequence,
                    "loss_mask": loss_mask,
                    "input_len": len(sequence) + len(self.separator),
                }
            )
        return Dataset.from_pandas(pd.DataFrame(data))


def generate_all_datasets(num_samples: int = 1000000) -> Dict[str, Dataset]:
    """Generate all synthetic datasets."""
    generator = SyntheticDataGenerator()

    datasets = {
        "copy_task": generator.generate_copy_task(num_samples),
        "reverse_task": generator.generate_reverse_task(num_samples),
        "sort_task": generator.generate_sort_task(num_samples),
        "arithmetic_task": generator.generate_arithmetic_task(num_samples),
        "pattern_completion_task": generator.generate_pattern_completion_task(
            num_samples
        ),
        "bracket_matching_task": generator.generate_bracket_matching_task(num_samples),
    }

    return datasets


# Example usage:
if __name__ == "__main__":
    load_dotenv()

    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    login(token=hf_token)

    # Generate datasets with 1M samples each
    datasets = generate_all_datasets(1000000)

    # Push to Hub (you'll need to be logged in)
    for name, dataset in datasets.items():
        dataset.push_to_hub(f"flaitenberger/synthetic_{name}")
