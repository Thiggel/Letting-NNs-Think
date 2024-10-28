from experiment.experiment import Experiment
from experiment.configs import TrainingConfig, DataConfig, ModelConfig, EvaluationConfig
from experiment.runners import ExperimentRunner

experiment = Experiment(
    config_classes=[ModelConfig, TrainingConfig, DataConfig, EvaluationConfig],
    runner_class=ExperimentRunner,
)


@experiment.cli()
def main(exp: Experiment):
    """
    Main function to run the experiment.
    """
    # Access configs through exp.configs
    results = exp.run()
    return results


if __name__ == "__main__":
    # Run the Typer application by calling experiment.run()
    experiment.run()
