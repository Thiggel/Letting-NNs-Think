from dataclasses import dataclass
from pathlib import Path
import wandb
import os
import time
import torch
from typing import Type, TypeVar, Dict, List, Optional
from numbers import Number
from dotenv import load_dotenv
from huggingface_hub import login
from pydantic import BaseModel
from logging import getLogger, Logger

from cli_manager import CLIManager

from .Runner import Runner
from .ExperimentConfig import ExperimentConfig


T = TypeVar("T", bound=BaseModel)
ResultType = Dict[str, float]


@dataclass
class ExperimentResult:
    """Container for experiment results with metrics and metadata"""

    metrics: ResultType
    training_time: float
    seed: int
    run_idx: int


class Experiment:
    """
    A class to manage multiple runs of an experiment with different seeds,
    handling configuration, logging, and result aggregation.
    """

    def __init__(
        self,
        config_classes: List[Type[BaseModel]],
        runner_class: Type[Runner],
    ):
        """
        Initialize experiment with configurations and runner.

        Args:
            config_classes: List of Pydantic model classes for configuration
            runner_class: Class responsible for running individual experiments
        """
        self.logger = self._setup_logger()
        self.configs = self._initialize_experiment(config_classes)
        self.experiment_config: ExperimentConfig = self.configs[
            ExperimentConfig.__name__
        ]
        self.runner_class = runner_class
        self.log_dir = Path(os.getenv("BASE_CACHE_DIR") or "") / "logs" or Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Validate configuration
        self._validate_experiment_setup()

    def _setup_logger(self) -> Logger:
        """Set up logging for the experiment"""
        logger = getLogger(__name__)
        # Add logging configuration here if needed
        return logger

    def _initialize_experiment(
        self, config_classes: List[Type[BaseModel]]
    ) -> Dict[str, BaseModel]:
        """Initialize experiment configuration and external services"""
        load_dotenv()

        # Parse command line arguments
        configs = self._parse_args(config_classes)

        # Initialize external services
        self._setup_external_services()

        return configs

    def _setup_external_services(self) -> None:
        """Set up external services like HuggingFace and WandB"""
        try:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                login(token=hf_token)
            else:
                self.logger.warning("No HuggingFace token found in environment")

            if self.experiment_config.enable_logging:
                wandb_key = os.getenv("WANDB_API_KEY")
                if wandb_key:
                    wandb.login(key=wandb_key)
                else:
                    self.logger.warning("No WandB API key found in environment")
        except Exception as e:
            self.logger.error(f"Failed to initialize external services: {str(e)}")
            raise

    def _parse_args(
        self, config_classes: List[Type[BaseModel]]
    ) -> Dict[str, BaseModel]:
        """Parse command line arguments using CLIManager"""
        cli_manager = CLIManager(*[*config_classes, ExperimentConfig])
        cli_manager.register_command()
        cli_manager.run()
        return cli_manager.get_all_configs()

    def _validate_experiment_setup(self) -> None:
        """Validate experiment configuration"""
        if self.experiment_config.num_runs != len(self.experiment_config.seeds):
            raise ValueError(
                f"Number of runs ({self.experiment_config.num_runs}) must match "
                f"number of seeds ({len(self.experiment_config.seeds)})"
            )

    def run(self) -> Dict[str, Dict[str, Number]]:
        """
        Run the complete experiment with multiple seeds and return aggregated results.

        Returns:
            Dictionary containing mean and standard deviation for each metric
        """
        try:
            self.logger.info(
                f"Starting experiment: {self.experiment_config.experiment_name}"
            )
            all_results = self.run_different_seeds()
            aggregated_results = self.get_mean_std(all_results)

            self.print_mean_std(aggregated_results)
            self._save_results(all_results, aggregated_results)

            return aggregated_results
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise

    def run_different_seeds(self) -> List[ExperimentResult]:
        """Run experiment with different seeds and collect results"""
        all_results = []

        for run_idx in range(self.experiment_config.num_runs):
            seed = self.experiment_config.seeds[run_idx]
            self.logger.info(
                f"Starting run {run_idx + 1}/{self.experiment_config.num_runs} with seed {seed}"
            )

            try:
                result = self._run_single_experiment(run_idx, seed)
                all_results.append(result)
                self.logger.info(f"Run {run_idx + 1} completed successfully")
            except Exception as e:
                self.logger.error(f"Run {run_idx + 1} failed: {str(e)}")
                if not self.experiment_config.continue_on_error:
                    raise

        return all_results

    def _run_single_experiment(self, run_idx: int, seed: int) -> ExperimentResult:
        """Run a single experiment with given seed"""
        start_time = time.time()

        runner = self.runner_class(self.configs)
        metrics = runner.run(seed=seed)

        training_time = (time.time() - start_time) / 3600  # Convert to hours

        return ExperimentResult(
            metrics=metrics, training_time=training_time, seed=seed, run_idx=run_idx
        )

    def get_mean_std(
        self, results: List[ExperimentResult]
    ) -> Dict[str, Dict[str, Number]]:
        """Calculate mean and standard deviation of results"""
        metrics_list = [result.metrics for result in results]
        tensor_data = torch.tensor([list(d.values()) for d in metrics_list])

        mean = tensor_data.mean(dim=0)
        std_dev = tensor_data.std(dim=0)

        return {
            str(key): {"mean": m.item(), "std": s.item()}
            for key, m, s in zip(metrics_list[0].keys(), mean, std_dev)
        }

    def print_mean_std(self, mean_std: Dict[str, Dict[str, Number]]) -> None:
        """Print aggregated results"""
        self.logger.info("Final Results:")
        for key, values in mean_std.items():
            self.logger.info(
                f"{key} - Mean: {values['mean']:.4f}, Std: {values['std']:.4f}"
            )

    def _save_results(
        self,
        all_results: List[ExperimentResult],
        aggregated_results: Dict[str, Dict[str, Number]],
    ) -> None:
        """Save experiment results to disk"""
        if self.experiment_config.save_results:
            save_path = (
                self.log_dir / f"{self.experiment_config.experiment_name}_results.pt"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "individual_results": all_results,
                    "aggregated_results": aggregated_results,
                    "config": self.configs,
                },
                save_path,
            )

            self.logger.info(f"Results saved to {save_path}")
