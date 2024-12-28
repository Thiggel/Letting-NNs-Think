from torch.utils.data import IterableDataset
import numpy as np


class ArithmeticDataset(IterableDataset):
    def __init__(
        self, max_len: int = 50, min_len: int = 3, tokenizer=None, process_fn=None
    ):
        self.max_len = max_len
        self.min_len = min_len
        self.tokenizer = tokenizer
        self.process_fn = process_fn

    def __iter__(self):
        while True:
            length = np.random.randint(self.min_len, self.max_len)
            sequence = np.random.randint(0, 10, size=length)

            expression = " + ".join(map(str, sequence))
            result = sum(sequence)

            text = f"{expression} = {result} [EOS]"

            if self.process_fn is not None:
                yield self.process_fn({"text": text})
            else:
                yield {"text": text}
