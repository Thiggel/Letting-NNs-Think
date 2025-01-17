from typing import Any
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from pydantic import BaseModel
import wandb

from experiment.runners.EvaluationRunner import EvaluationRunner


class RecursiveCurriculumCallback(Callback):
    """Implements curriculum learning for recursive transformers by gradually increasing num_steps."""

    def __init__(
        self,
        configs: dict[str, BaseModel],
        seed: int,
        start_steps: int = 1,
        max_steps: int = 20,
        convergence_window: int = 5,
        min_delta: float = 0.01,
        patience: int = 3,
    ):
        """
        Args:
            configs: Dictionary of all configuration objects
            start_steps: Initial number of recursive steps
            max_steps: Maximum number of recursive steps to reach
            convergence_window: Number of epochs to average validation loss over
            min_delta: Minimum change in validation loss to consider non-converged
            patience: Number of convergence windows to wait before increasing steps
        """
        super().__init__()
        self.configs = configs
        self.seed = seed
        self.current_steps = start_steps
        self.max_steps = max_steps
        self.convergence_window = convergence_window
        self.min_delta = min_delta
        self.patience = patience

        # Evaluation runner
        self.evaluation_runner = EvaluationRunner(configs)

        # State tracking
        self.val_losses: list[float] = []
        self.converged_windows = 0
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
        """Check for convergence after each validation epoch."""
        if not trainer.sanity_checking:
            current_loss = trainer.callback_metrics.get("val_loss")
            if current_loss is None:
                return

            # Convert to Python float for easier handling
            current_loss = (
                current_loss.item()
                if isinstance(current_loss, torch.Tensor)
                else float(current_loss)
            )

            # Add to rolling window
            self.val_losses.append(current_loss)
            if len(self.val_losses) > self.convergence_window:
                self.val_losses.pop(0)

            # Only check convergence if we have enough samples
            if len(self.val_losses) == self.convergence_window:
                self._check_convergence(pl_module)

    def _check_convergence(self, pl_module: Any) -> None:
        """Check if training has converged and increase steps if needed."""
        # Calculate average loss over window
        avg_loss = sum(self.val_losses) / len(self.val_losses)

        # Check if loss has improved significantly
        relative_improvement = abs(self.best_val_loss - avg_loss) / abs(
            self.best_val_loss
        )

        if relative_improvement < self.min_delta:
            self.converged_windows += 1
            print(f"Converged for {self.converged_windows}/{self.patience} windows")
        else:
            self.converged_windows = 0
            self.best_val_loss = min(self.best_val_loss, avg_loss)
            print(
                f"Improved by {relative_improvement:.2f} (> delta of {self.min_delta}) to {avg_loss:.2f}"
            )

        # If converged for enough windows and not at max steps, increase steps
        if (
            self.converged_windows >= self.patience
            and self.current_steps < self.max_steps
        ):
            self._increase_steps(pl_module)

    def _evaluate_current_model(self, pl_module: Any) -> dict:
        """Run evaluation on the current model state."""
        pl_module.eval()

        self.evaluation_runner.evaluation_config.load_from_checkpoint = None

        self.evaluation_runner.experiment_config.enable_logging = False

        current_state = pl_module.state_dict()
        results = self.evaluation_runner.run(self.seed, state_dict=current_state)

        results["recursive_steps"] = self.current_steps

        # Log to wandb if enabled
        if wandb.run is not None:
            wandb.log(results)

        pl_module.train()
        return results

    def _increase_steps(self, pl_module: Any) -> None:
        """Evaluate current model and increase the number of recursive steps."""
        # First evaluate current model
        print(f"\nEvaluating model at {self.current_steps} steps...")
        evaluation_results = self._evaluate_current_model(pl_module)
        print(f"Evaluation results: {evaluation_results}")

        # Then increase steps
        self.current_steps += 1
        print(f"Increasing recursive steps to {self.current_steps}")

        # Update model's num_steps
        recurrent_layer = pl_module.get_recurrent_layer()
        if recurrent_layer:
            self.configs["ModelConfig"].num_steps = self.current_steps
            recurrent_layer.strategy.num_steps = self.current_steps

        # Reset convergence tracking
        self.val_losses = []
        self.converged_windows = 0
        self.best_val_loss = float("inf")
