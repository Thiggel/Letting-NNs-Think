from torch.utils.data import IterableDataset
import numpy as np


class InfinitePatternDataset(IterableDataset):
    def __init__(self, seq_length: int = 5):
        self.seq_length = seq_length

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
            yield {
                "text": text,
                "loss_mask": [0] * (len(input_seq) + 4) + [1] * len(target),
                "input_len": len(input_seq) + 4,  # +4 for " -> "
            }
