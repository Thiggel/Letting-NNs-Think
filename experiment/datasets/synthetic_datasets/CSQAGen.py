import os
import json
from typing import Optional, Callable
from torch.utils.data import Dataset


class CSQAGen(Dataset):
    def __init__(
        self,
        root_dir: str = "csqa_gen",
        tokenizer=None,
        process_fn: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.process_fn = process_fn

        self.files = [
            os.path.join(self.root_dir, file)
            for file in os.listdir(self.root_dir)
            if file.endswith(".json")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, str]:
        with open(self.files[index], "r") as f:
            data = json.load(f)

        query = data["query"]
        steps = "\n".join(data["steps"])
        solution = data["solution"]

        if self.process_fn is not None:
            return self.process_fn(
                {"query": query, "steps": steps, "solution": solution}
            )

        return {"query": query, "steps": steps, "solution": solution}
