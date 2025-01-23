from torch.utils.data import IterableDataset
import numpy as np


class PatternDataset(IterableDataset):
    def __init__(self, seq_length: int = 5, tokenizer=None, process_fn=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.process_fn = process_fn

    def __iter__(self):
        while True:
            start = np.random.randint(-100, 100)
            step = np.random.randint(-20, 20)
            if step == 0:
                step = 1

            sequence = [start + i * step for i in range(self.seq_length)]
            input_seq = " ".join(map(str, sequence[:-1]))
            target = str(sequence[-1])

            text = f"{input_seq} -> {target}"

            if self.process_fn is not None:
                yield self.process_fn(
                    {
                        "text": text,
                        "loss_mask": [0] * (len(input_seq) + 4) + [1] * len(target),
                    }
                )
            else:
                yield {
                    "text": text,
                    "loss_mask": [0] * (len(input_seq) + 4) + [1] * len(target),
                }
