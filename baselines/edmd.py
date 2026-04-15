"""
Extended Dynamic Mode Decomposition (EDMD) baselines.

Supports polynomial and radial-basis-function dictionaries.
Fits the Koopman matrix K via least-squares on lifted data.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Literal, Optional

import numpy as np
from scipy.linalg import lstsq


class EDMDModel:
    """EDMD with a configurable observable dictionary."""

    def __init__(
        self,
        dictionary: Literal["poly", "rbf"] = "poly",
        poly_degree: int = 4,
        n_rbf: int = 100,
        rbf_sigma: float = 1.0,
    ):
        self.dictionary = dictionary
        self.poly_degree = poly_degree
        self.n_rbf = n_rbf
        self.rbf_sigma = rbf_sigma

        self.K: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self._rbf_centers: Optional[np.ndarray] = None

    def _build_poly_features(self, X: np.ndarray) -> np.ndarray:
        """Build monomial features up to poly_degree."""
        n_samples, n_features = X.shape
        cols = [np.ones((n_samples, 1))]
        for deg in range(1, self.poly_degree + 1):
            for combo in combinations_with_replacement(range(n_features), deg):
                col = np.ones(n_samples)
                for idx in combo:
                    col = col * X[:, idx]
                cols.append(col.reshape(-1, 1))
        return np.hstack(cols)

    def _build_rbf_features(self, X: np.ndarray) -> np.ndarray:
        """Build RBF features using random centers."""
        diffs = X[:, np.newaxis, :] - self._rbf_centers[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=-1)
        return np.exp(-dists_sq / (2 * self.rbf_sigma ** 2))

    def lift(self, X: np.ndarray) -> np.ndarray:
        if self.dictionary == "poly":
            return self._build_poly_features(X)
        return self._build_rbf_features(X)

    def fit(self, X_k: np.ndarray, X_k1: np.ndarray) -> "EDMDModel":
        if self.dictionary == "rbf":
            idx = np.random.choice(len(X_k), min(self.n_rbf, len(X_k)), replace=False)
            self._rbf_centers = X_k[idx]

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
