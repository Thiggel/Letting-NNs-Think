import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from typing import Callable


class ThresholdOptimizer:
    """
    Active‐learning GP for a single scalar threshold:
      - evaluate_fn: t -> compute_saved (float)
      - sequentially sample t where GP uncertainty is highest
      - invert final GP mean to get threshold for any target savings s
    """

    def __init__(
        self,
        evaluate_fn: Callable[[float], float],
        initial_samples: int = 5,
        grid_size: int = 1000,
    ):
        self.evaluate_fn = evaluate_fn
        # a fixed grid for acquisition (1D)
        self._grid = np.linspace(0.0, 1.0, grid_size).reshape(-1, 1)
        # data
        self.X = np.empty((0, 1))
        self.y = np.empty((0,))
        # GP with a smooth Matérn + tiny noise
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def initialize(self):
        # random initial points
        xs = np.random.rand(self.initial_samples, 1)
        ys = np.array([self.evaluate_fn(float(x)) for x in xs])
        self.X, self.y = xs, ys
        self.gp.fit(self.X, self.y)

    def propose(self) -> float:
        # predict posterior std on the grid, pick argmax
        _, sigma = self.gp.predict(self._grid, return_std=True)
        idx = np.argmax(sigma)
        return float(self._grid[idx, 0])

    def update(self, t: float):
        # evaluate and refit
        c = self.evaluate_fn(t)
        self.X = np.vstack([self.X, [[t]]])
        self.y = np.append(self.y, c)
        self.gp.fit(self.X, self.y)

    def run(self, iterations: int = 20):
        self.initialize()
        for _ in range(iterations):
            t_next = self.propose()
            self.update(t_next)

    def get_threshold_for_s(self, s: float, tol: float = 1e-3) -> float:
        """
        Solve GP_mean(t) = s by bisection on [0,1].
        If no sign change, returns closest endpoint.
        """

        def f(t):
            return float(self.gp.predict([[t]])) - s

        a, b = 0.0, 1.0
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            # no root in [0,1]: pick closer end
            return a if abs(fa) < abs(fb) else b

        for _ in range(30):
            m = 0.5 * (a + b)
            fm = f(m)
            if abs(fm) < tol:
                return m
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)
