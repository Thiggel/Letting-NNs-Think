from pathlib import Path
import json
from tqdm import tqdm

samples = Path("../../../../../projects/prjs1147/flaitenberger/samples")


def load_sample(sample):
    with open(sample, "r") as f:
        return clean_samples(json.load(f))


def clean_samples(samples):
    correct_samples = {}
    incorrect_samples = {}

    for metric in samples:
        correct_samples[metric] = []
        incorrect_samples[metric] = []

        for sample in samples[metric]:
            cleaned_sample = {
                "input": sample["prompt_hash"],
                "output": sample["doc_hash"],
            }

            if sample["acc"] == 1.0:
                correct_samples[metric].append(cleaned_sample)
            else:
                incorrect_samples[metric].append(cleaned_sample)

    return {
        "correct": correct_samples,
        "incorrect": incorrect_samples,
    }


all_samples = {
    sample.name: load_sample(sample)
    for sample in tqdm(samples.iterdir())
    if sample.suffix == ".json"
}

print(all_samples)
