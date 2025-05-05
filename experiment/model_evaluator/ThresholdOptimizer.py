import torch
from torch import Tensor
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.acquisition.objective import IdentityMCMultiOutput
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from typing import Callable, List, Tuple


class ThresholdOptimizer:
    """
    Bayesian optimizer for layer-wise gating thresholds.

    Uses a multi-output GP to model (compute_saved, accuracy) as functions of
    per-layer thresholds, and sequentially samples new threshold vectors
    via qEHVI to build an approximate Pareto front.
    """
    def __init__(
        self,
        evaluate_fn: Callable[[Tensor], Tuple[float, float]],
        num_layers: int,
        initial_samples: int = 5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            evaluate_fn: function mapping a tensor of shape (..., num_layers)
                to a tuple (compute_saved, accuracy).
            num_layers: number of gating layers (dimension of threshold vector).
            initial_samples: number of initial random samples.
            device: torch.device.
            dtype: torch.dtype.
        """
        self.evaluate_fn = evaluate_fn
        self.num_layers = num_layers
        self.initial_samples = initial_samples
        self.device = device
        self.dtype = dtype

        # storage for past evaluations
        self.train_x = torch.empty(0, num_layers, device=device, dtype=dtype)
        # train_y: columns are [compute_saved, accuracy]
        self.train_y = torch.empty(0, 2, device=device, dtype=dtype)

        self.model: ModelListGP = None
        self.ref_point = None  # for hypervolume

    def initialize(self) -> None:
        """Draw initial samples via Sobol, evaluate them."""
        sobol = torch.quasirandom.SobolEngine(dimension=self.num_layers, scramble=True)
        x_init = sobol.draw(self.initial_samples).to(self.device, self.dtype)
        ys = [self._evaluate(x_init[i]) for i in range(self.initial_samples)]
        y_init = torch.tensor(ys, device=self.device, dtype=self.dtype)
        self.train_x = x_init
        self.train_y = y_init

    def _evaluate(self, x: Tensor) -> Tuple[float, float]:
        """Evaluate compute_saved and accuracy for a single threshold vector."""
        # x: 1D tensor of length num_layers
        c, a = self.evaluate_fn(x)
        return float(c), float(a)

    def _fit_model(self) -> None:
        """Fit two independent GPs and wrap in a ModelListGP."""
        # GP for compute_saved
        gp_c = SingleTaskGP(self.train_x, self.train_y[:, :1])
        mll_c = ExactMarginalLogLikelihood(gp_c.likelihood, gp_c)
        fit_gpytorch_model(mll_c)
        # GP for accuracy
        gp_a = SingleTaskGP(self.train_x, self.train_y[:, 1:2])
        mll_a = ExactMarginalLogLikelihood(gp_a.likelihood, gp_a)
        fit_gpytorch_model(mll_a)
        # wrap
        self.model = ModelListGP(gp_c, gp_a)

        # reference point: slightly below worst observed
        worst = torch.min(self.train_y, dim=0).values
        self.ref_point = (worst - 0.1).tolist()

    def _propose_candidate(self) -> Tensor:
        """Use qEHVI to propose the next threshold vector."""
        sampler = SobolQMCNormalSampler(num_samples=128)
        acq = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.ref_point,
            partitioning=None,
            sampler=sampler,
            objective=IdentityMCMultiOutput()
        )
        bounds = torch.stack([
            torch.zeros(self.num_layers, device=self.device, dtype=self.dtype),
            torch.ones(self.num_layers, device=self.device, dtype=self.dtype)
        ])
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        return candidate.detach().squeeze(0)

    def run(self, iterations: int = 20) -> None:
        """
        Main loop: initialize, then for a number of iterations: fit surrogate,
        propose, evaluate, and update dataset.
        """
        # 1) initial design
        self.initialize()
        # 2) sequential loop
        for _ in range(iterations):
            # fit GPs
            self._fit_model()
            # propose next point
            x_new = self._propose_candidate()
            # evaluate
            y_c, y_a = self.evaluate_fn(x_new)
            y_new = torch.tensor([[y_c, y_a]], device=self.device, dtype=self.dtype)
            # append
            self.train_x = torch.cat([self.train_x, x_new.unsqueeze(0)], dim=0)
            self.train_y = torch.cat([self.train_y, y_new], dim=0)

    def get_thresholds_for_s(self, s: float) -> List[float]:
        """
        Given a desired compute_saved s (0 < s < 1), return the threshold vector
        whose evaluated compute_saved is closest to s.
        """
        diffs = torch.abs(self.train_y[:, 0] - s)
        idx = torch.argmin(diffs).item()
        return self.train_x[idx].tolist()
