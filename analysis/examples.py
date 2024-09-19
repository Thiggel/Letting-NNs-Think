from pathlib import Path
import json
from tqdm import tqdm

samples = Path("../../../../../projects/prjs1147/flaitenberger/samples/default")


def load_sample(sample):
    with open(sample, "r") as f:
        return clean_samples(json.load(f))


def format_output(output):
    if len(output) == 1:
        return output[0]

    try:
        max = 0
        max_index = 0
        for index, option in enumerate(output[0]):
            if option[0] > max:
                max = option[0]
                max_index = index
    except Exception as e:
        print(output)
        exit()

    return ["A", "B", "C", "D", "E", "F", "G"][max_index]


def clean_samples(samples):
    correct_samples = {}
    incorrect_samples = {}

    for metric in samples:
        correct_samples[metric] = []
        incorrect_samples[metric] = []

        for sample in samples[metric]:
            cleaned_sample = {
                "input": sample["arguments"][0],
                "output": format_output(sample["resps"]),
            }

            if ("acc" in sample and sample["acc"] == 1.0) or (
                "exact_match" in sample and sample["exact_match"] == 1.0
            ):
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


# for each sample file (which corresponds to the sample of one model)
# and for each metric, find a sample that this model got correct but all
# other models got wrong, and vice versa
for sample in all_samples:
    for metric in all_samples[sample]["correct"]:
        for index, correct_sample in enumerate(all_samples[sample]["correct"][metric]):
            other_answers = {}
            num_other_answers = 0
            for other_sample in all_samples:
                if other_sample == sample:
                    continue

                try:
                    other_answer = next(
                        incorrect_sample
                        for incorrect_sample in all_samples[other_sample]["incorrect"][
                            metric
                        ]
                        if incorrect_sample["input"] == correct_sample["input"]
                    )["output"]

                    other_answers[other_sample] = other_answer
                    num_other_answers += 1
                except StopIteration:
                    pass

            all_samples[sample]["correct"][metric][index]["other_answers"] = other_answers
            all_samples[sample]["correct"][metric][index][
                "num_other_answers"
            ] = num_other_answers

        all_samples[sample]["correct"][metric] = sorted(
            all_samples[sample]["correct"][metric],
            key=lambda x: x["num_other_answers"],
            reverse=True,
        )


for sample in all_samples:
    for metric in all_samples[sample]["correct"]:
        if len(all_samples[sample]["correct"][metric]) == 0:
            continue

        correct_sample = all_samples[sample]["correct"][metric][0]

        print(
            f"{sample} got correct sample that many other models got wrong on {metric}"
        )
        print(correct_sample["input"])
        print(correct_sample["output"])
        print("\nOther answers:")
        other_answers = correct_sample["other_answers"]
        for other_sample in other_answers:
            print(f"{other_sample}: {other_answers[other_sample]}")
        print("\n\n\n")
