import torch
from torch import Tensor
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
import numpy as np
from typing import Callable, List, Tuple
from tqdm import tqdm
import os
import sys

from experiment.utils.suppress_output import suppress_all_output

class ThresholdOptimizer:
    """
    Bayesian optimizer for layer-wise gating thresholds.
    Models (compute_saved, accuracy) with multi-output GP,
    uses qEHVI sequential sampling to approximate the Pareto front.
    Provides f(s) by interpolating on the non-dominated threshold vectors.
    """
    def __init__(
        self,
        evaluate_fn: Callable[[torch.Tensor], Tuple[float, float]],
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
        self.model: ModelListGP = None
        self.ref_point: List[float] = None

    def initialize(self) -> None:
        sobol = torch.quasirandom.SobolEngine(self.num_layers, scramble=True)
        x_init = sobol.draw(self.initial_samples).to(self.device, self.dtype)
        ys = []
        for x in tqdm(x_init, desc="Initial Sampling", leave=False):
            with suppress_all_output():
                ys.append(self.evaluate_fn(x))
        y_init = torch.tensor(ys, device=self.device, dtype=self.dtype)
        self.train_x, self.train_y = x_init, y_init

    def _fit_model(self) -> None:
        # GP for compute_saved
        gp_c = SingleTaskGP(self.train_x, self.train_y[:, :1])
        mll_c = ExactMarginalLogLikelihood(gp_c.likelihood, gp_c)
        fit_gpytorch_mll(mll_c)
        # GP for accuracy
        gp_a = SingleTaskGP(self.train_x, self.train_y[:, 1:2])
        mll_a = ExactMarginalLogLikelihood(gp_a.likelihood, gp_a)
        fit_gpytorch_mll(mll_a)
        # combine
        self.model = ModelListGP(gp_c, gp_a)
        worst = torch.min(self.train_y, dim=0).values
        self.ref_point = (worst - 0.1).tolist()

    def _propose_candidate(self) -> torch.Tensor:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        partitioning = NondominatedPartitioning(
            ref_point=torch.tensor(self.ref_point, device=self.device, dtype=self.dtype),
            Y=self.train_y,
        )
        acq = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.ref_point,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            partitioning=partitioning,
        )
        bounds = torch.stack([
            torch.zeros(self.num_layers, device=self.device, dtype=self.dtype),
            torch.ones(self.num_layers, device=self.device, dtype=self.dtype),
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
        Build up training data via qEHVI sampling.
        After this, self.train_x and train_y contain all evaluated points.
        """
        self.initialize()
        for _ in tqdm(range(iterations), desc="Threshold Optimization", leave=False, unit="it"):
            with suppress_all_output():
                self._fit_model()
                x_new = self._propose_candidate()
                y_new = torch.tensor(self.evaluate_fn(x_new), device=self.device, dtype=self.dtype)
                self.train_x = torch.cat([self.train_x, x_new.unsqueeze(0)], dim=0)
                self.train_y = torch.cat([self.train_y, y_new.unsqueeze(0)], dim=0)

    def pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract non-dominated (compute_saved, accuracy) points
        and corresponding thresholds. Returns sorted arrays:
        s_vals (compute_saved), t_vals (threshold vectors).
        """
        xs = self.train_x.cpu().numpy()  # shape (M, N)
        ys = self.train_y.cpu().numpy()  # shape (M, 2)
        # non-domination: nobody strictly better in both dims
        mask = np.ones(len(ys), dtype=bool)
        for i in range(len(ys)):
            for j in range(len(ys)):
                if (ys[j, 0] >= ys[i, 0] and ys[j, 1] >= ys[i, 1]) and (
                   ys[j, 0] > ys[i, 0] or ys[j, 1] > ys[i, 1]):
                    mask[i] = False
                    break
        pareto_x = xs[mask]
        pareto_y = ys[mask]
        # sort by compute_saved
        idx = np.argsort(pareto_y[:, 0])
        return pareto_y[idx, 0], pareto_x[idx]

    def get_thresholds_for_s(self, s: float) -> List[float]:
        """
        Interpolate on the Pareto front to get thresholds for any s in [min, max].
        """
        s_arr, t_arr = self.pareto_front()
        # clamp
        if s <= s_arr[0]:
            return t_arr[0].tolist()
        if s >= s_arr[-1]:
            return t_arr[-1].tolist()
        # find segment
        for i in range(len(s_arr) - 1):
            if s_arr[i] <= s <= s_arr[i + 1]:
                alpha = (s - s_arr[i]) / (s_arr[i + 1] - s_arr[i])
                t = (1 - alpha) * t_arr[i] + alpha * t_arr[i + 1]
                return t.tolist()
        # fallback nearest
        idx = np.abs(s_arr - s).argmin()
        return t_arr[idx].tolist()
