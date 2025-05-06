import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from typing import Callable


class ThresholdOptimizer:
    """
    Active‐learning on a single scalar threshold t∈[0,1] for two outputs:
     - compute_saved c(t)
     - retention r(t)=accuracy(t)/accuracy(0)
    We fit two independent GPs and at each iteration pick the t
    maximizing sigma_c(t)+sigma_r(t).
    """
    def __init__(
        self,
        evaluate_fn: Callable[[float], Tuple[float, float]],
        initial_samples: int = 5,
        grid_size: int = 500,
    ):
        self.evaluate_fn = evaluate_fn
        self.grid = np.linspace(0.0, 1.0, grid_size).reshape(-1, 1)
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp_c = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gp_r = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.X = np.empty((0, 1))
        self.y_c = np.empty((0,))
        self.y_r = np.empty((0,))

    def initialize(self):
        # Random seed points
        xs = np.random.rand(self.initial_samples, 1)
        outs = np.array([self.evaluate_fn(x[0]) for x in xs])
        self.X = xs
        self.y_c = outs[:, 0]
        self.y_r = outs[:, 1]
        self.gp_c.fit(self.X, self.y_c)
        self.gp_r.fit(self.X, self.y_r)

    def propose(self) -> float:
        # Posterior std on grid
        _, s_c = self.gp_c.predict(self.grid, return_std=True)
        _, s_r = self.gp_r.predict(self.grid, return_std=True)
        scores = s_c + s_r
        idx = np.argmax(scores)
        return float(self.grid[idx, 0])

    def update(self, t: float):
        c, r = self.evaluate_fn(t)
        self.X = np.vstack([self.X, [[t]]])
        self.y_c = np.append(self.y_c, c)
        self.y_r = np.append(self.y_r, r)
        self.gp_c.fit(self.X, self.y_c)
        self.gp_r.fit(self.X, self.y_r)

    def run(self, iterations: int = 20):
        self.initialize()
        for _ in range(iterations):
            t_next = self.propose()
            self.update(t_next)

    def _invert_gp(self, gp: GaussianProcessRegressor, target: float, tol: float = 1e-3) -> float:
        # bisection solve gp.mean(t)=target on [0,1]
        def f(x):
            return gp.predict([[x]])[0] - target
        a, b = 0.0, 1.0
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            return a if abs(fa) < abs(fb) else b
        for _ in range(30):
            m = 0.5*(a+b)
            fm = f(m)
            if abs(fm) < tol:
                return m
            if fa*fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5*(a+b)

    def get_threshold_for_compute(self, s: float) -> float:
        """Return t so that compute_saved≈s."""
        return self._invert_gp(self.gp_c, s)

    def get_threshold_for_retention(self, s: float) -> float:
        """Return t so that retention≈s (e.g. s=0.9 for 90%)."""
        return self._invert_gp(self.gp_r, s) return 0.5 * (a + b)
