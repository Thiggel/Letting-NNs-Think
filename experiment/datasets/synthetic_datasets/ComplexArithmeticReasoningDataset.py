from torch.utils.data import IterableDataset
import numpy as np
import random


class ComplexArithmeticReasoningDataset(IterableDataset):
    def __init__(
        self, max_len: int = 15, min_len: int = 8, tokenizer=None, process_fn=None
    ):
        self.max_len = max_len
        self.min_len = min_len
        self.tokenizer = tokenizer
        self.process_fn = process_fn
        self.operators = ["+", "-", "*"]

    def generate_expression(self, length):
        # Generate numbers between 0 and 9
        numbers = [str(random.randint(0, 9)) for _ in range(length)]
        # Generate operators with some multiplication
        operators = []
        for _ in range(length - 1):
            # Higher weight for addition/subtraction to avoid too complex calculations
            weights = [0.4, 0.4, 0.2]  # +, -, *
            operators.append(np.random.choice(self.operators, p=weights))

        # Create expression
        expression = ""
        for i in range(length - 1):
            expression += numbers[i] + operators[i]
        expression += numbers[-1]

        return expression

    def evaluate_with_steps(self, expression):
        terms = expression.split("+")
        new_terms = []
        steps = []

        for term in terms:
            sub_terms = term.split("-")
            processed_sub_terms = []

            for sub_term in sub_terms:
                if "*" in sub_term:
                    mult_terms = sub_term.split("*")
                    result = int(mult_terms[0])
                    sub_steps = []

                    for i in range(1, len(mult_terms)):
                        prev_result = result
                        result *= int(mult_terms[i])
                        sub_steps.append(f"{prev_result}*{mult_terms[i]}={result}")

                    steps.append(f"({' -> '.join(sub_steps)})")
                    processed_sub_terms.append(str(result))
                else:
                    processed_sub_terms.append(sub_term)

            reconstructed_term = processed_sub_terms[0]
            current_value = int(processed_sub_terms[0])

            for i in range(1, len(processed_sub_terms)):
                subtrahend = int(processed_sub_terms[i])
                steps.append(
                    f"({current_value}-{subtrahend}={current_value-subtrahend})"
                )
                current_value -= subtrahend
                reconstructed_term = str(current_value)

            new_terms.append(reconstructed_term)

        result = int(new_terms[0])
        for i in range(1, len(new_terms)):
            addend = int(new_terms[i])
            steps.append(f"({result}+{addend}={result+addend})")
            result += addend

        steps.append(f" Result: {result}")

        return steps

    def __iter__(self):
        while True:
            length = random.randint(self.min_len, self.max_len)
            expression = self.generate_expression(length)
            steps = self.evaluate_with_steps(expression)
            text = f"Query:{expression} Answer:{''.join(steps)}"

            if self.process_fn is not None:
                yield self.process_fn({"text": text})
            else:
                yield {"text": text}
