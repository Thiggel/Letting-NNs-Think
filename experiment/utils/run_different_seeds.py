import time

from experiment.utils import Args
from experiment.utils import run


def run_different_seeds(args: Args) -> list[dict]:
    all_results = []

    for run_idx in range(args.num_runs):
        start_time = time.time()

        results = run(args, seed=args.seeds[run_idx])

        end_time = time.time()
        seconds_to_hours = 3600
        training_time = (end_time - start_time) / seconds_to_hours
        results.update({"training_time": training_time})

        print(results)

        all_results.append(results)

    return all_results
