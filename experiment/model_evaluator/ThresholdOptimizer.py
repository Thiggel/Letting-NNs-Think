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
    Uses multi-output GPs (compute_saved, accuracy) with qEHVI.
    """

    def __init__(
        self,
        evaluate_fn,
        num_layers: int,
        initial_samples: int = 5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.evaluate_fn = evaluate_fn
        self.num_layers = num_layers
        self.initial_samples = initial_samples
        self.device = device
        self.dtype = dtype
        self.train_x = torch.empty(0, num_layers, device=device, dtype=dtype)
        self.train_y = torch.empty(0, 2, device=device, dtype=dtype)
        self.model = None
        self.ref_point = None

    def initialize(self) -> None:
        sobol = torch.quasirandom.SobolEngine(self.num_layers, scramble=True)
        x_init = sobol.draw(self.initial_samples).to(self.device, self.dtype)
        y_init = torch.stack(
            [
                torch.tensor(self.evaluate_fn(x), device=self.device, dtype=self.dtype)
                for x in x_init
            ],
            dim=0,
        )
        self.train_x, self.train_y = x_init, y_init

    def _fit_model(self) -> None:
        gp_c = SingleTaskGP(self.train_x, self.train_y[:, :1])
        mll_c = ExactMarginalLogLikelihood(gp_c.likelihood, gp_c)
        fit_gpytorch_model(mll_c)

        gp_a = SingleTaskGP(self.train_x, self.train_y[:, 1:2])
        mll_a = ExactMarginalLogLikelihood(gp_a.likelihood, gp_a)
        fit_gpytorch_model(mll_a)

        self.model = ModelListGP(gp_c, gp_a)
        worst = torch.min(self.train_y, dim=0).values
        self.ref_point = (worst - 0.1).tolist()

    def _propose_candidate(self) -> torch.Tensor:
        sampler = SobolQMCNormalSampler(num_samples=128)
        acq = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.ref_point,
            sampler=sampler,
            objective=IdentityMCMultiOutput(),
        )
        bounds = torch.stack(
            [
                torch.zeros(self.num_layers, device=self.device, dtype=self.dtype),
                torch.ones(self.num_layers, device=self.device, dtype=self.dtype),
            ]
        )
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        return candidate.detach().squeeze(0)

    def run(self, iterations: int = 20) -> None:
        self.initialize()
        for _ in range(iterations):
            self._fit_model()
            x_new = self._propose_candidate()
            y_new = torch.tensor(
                self.evaluate_fn(x_new), device=self.device, dtype=self.dtype
            )
            self.train_x = torch.cat([self.train_x, x_new.unsqueeze(0)], dim=0)
            self.train_y = torch.cat([self.train_y, y_new.unsqueeze(0)], dim=0)

    def get_thresholds_for_s(self, s: float) -> List[float]:
        diffs = torch.abs(self.train_y[:, 0] - s)
        idx = torch.argmin(diffs).item()
        return self.train_x[idx].tolist()
