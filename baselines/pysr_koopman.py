"""
PySR-based symbolic Koopman baseline.

Uses PySR to discover observable functions from trajectory data,
then fits a Koopman matrix K via least-squares on the discovered observables.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import lstsq

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except ImportError:
    _HAS_PYSR = False


class PySRKoopman:
    """Discover Koopman observables via PySR, then fit linear dynamics."""

    def __init__(
        self,
        n_observables: int = 16,
        niterations: int = 40,
        binary_operators: Optional[list[str]] = None,
        unary_operators: Optional[list[str]] = None,
    ):
        if not _HAS_PYSR:
            raise ImportError("PySR is required.  Install with: pip install pysr")

        self.n_observables = n_observables
        self.niterations = niterations
        self.binary_operators = binary_operators or ["+", "-", "*", "/"]
        self.unary_operators = unary_operators or ["exp", "log", "sin", "cos"]

        self._models: list[PySRRegressor] = []
        self._obs_fns: list = []
        self.K: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None

    def _discover_observables(self, X_k: np.ndarray, X_k1: np.ndarray) -> None:
        """Use PySR to find symbolic expressions that evolve linearly."""
        state_dim = X_k.shape[1]
        residuals = X_k1 - X_k

        for i in range(min(self.n_observables, state_dim)):
            model = PySRRegressor(
                niterations=self.niterations,
                binary_operators=self.binary_operators,
                unary_operators=self.unary_operators,
                populations=20,
                maxsize=20,
                verbosity=0,
            )
            model.fit(X_k, residuals[:, i % state_dim])
            self._models.append(model)

    def lift(self, X: np.ndarray) -> np.ndarray:
        cols = [np.ones((len(X), 1))]
        for col_i in range(X.shape[1]):
            cols.append(X[:, col_i:col_i + 1])
        for model in self._models:
            pred = model.predict(X).reshape(-1, 1)
            cols.append(pred)
        return np.hstack(cols)

    def fit(self, X_k: np.ndarray, X_k1: np.ndarray) -> "PySRKoopman":
        self._discover_observables(X_k, X_k1)
        G_k = self.lift(X_k)
        G_k1 = self.lift(X_k1)
        self.K, _, _, _ = lstsq(G_k, G_k1)
        self.C, _, _, _ = lstsq(G_k, X_k)
        return self

    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        g = self.lift(x0.reshape(1, -1))
        trajectory = [x0]
        for _ in range(n_steps):
            g = g @ self.K
            x_hat = g @ self.C
            trajectory.append(x_hat.flatten())
        return np.array(trajectory)
