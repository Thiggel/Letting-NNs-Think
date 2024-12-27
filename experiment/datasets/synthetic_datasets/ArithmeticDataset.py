from torch.utils.data import IterableDataset
import numpy as np


class ArithmeticDataset(IterableDataset):
    def __init__(self, max_len: int = 50, min_len: int = 3):
        self.max_len = max_len
        self.min_len = min_len

    def __iter__(self):
        while True:
            length = np.random.randint(self.min_len, self.max_len)
            sequence = np.random.randint(0, 100, size=length)

            expression = " + ".join(map(str, sequence))
            result = sum(sequence)

            text = f"{expression} = {result}"

            yield {
                "text": text,
                "input_len": len(expression) + 3,  # +3 for " = "
            }
