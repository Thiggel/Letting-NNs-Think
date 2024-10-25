from experiment.experiment import Experiment
from experiment.configs import TrainingConfig, DataConfig, ModelConfig, EvaluationConfig


if __name__ == "__main__":
    experiment = Experiment(
        config_classes=[ModelConfig, TrainingConfig, DataConfig, EvaluationConfig],
        runner_class=YourRunner,
    )

    results = experiment.run()
