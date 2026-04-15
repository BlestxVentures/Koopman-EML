"""
Full Koopman model with EML observable dictionary.

Architecture:
    g(x) = [T_1(x), ..., T_N(x)]   via EML trees   (lifting)
    g(x_{k+1}) ~ K @ g(x_k)                          (linear dynamics)
    x_hat = C @ g(x)                                  (reconstruction)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_eml.eml_tree import EMLTreeVectorized


class KoopmanEML(nn.Module):
    """Koopman operator model whose observable dictionary is a bank of EML trees."""

    def __init__(
        self,
        state_dim: int,
        n_observables: int = 32,
        tree_depth: int = 3,
        exp_order: int = 12,
        ln_order: int = 16,
        use_complex: bool = False,
        clamp_input: float = 5.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_obs = n_observables
        self.exp_order = exp_order
        self.ln_order = ln_order
        self.use_complex = use_complex
        self.clamp_input = clamp_input

        self.dictionary = EMLTreeVectorized(
            n_trees=n_observables, depth=tree_depth, n_vars=state_dim,
        )
        self.K = nn.Parameter(
            torch.eye(n_observables) + torch.randn(n_observables, n_observables) * 0.01
        )
        self.C = nn.Linear(n_observables, state_dim, bias=False)

    def lift(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Lift state to observable space: [B, state_dim] -> [B, n_obs]."""
        return self.dictionary(
            x, tau=tau, exp_order=self.exp_order, ln_order=self.ln_order,
            use_complex=self.use_complex, clamp_input=self.clamp_input,
        )

    def predict(self, g: torch.Tensor) -> torch.Tensor:
        """One-step prediction in observable space."""
        return g @ self.K.T

    def reconstruct(self, g: torch.Tensor) -> torch.Tensor:
        """Decode from observable space back to state space."""
        return self.C(g)

    def forward(self, x_k: torch.Tensor, x_k1: torch.Tensor, tau: float = 1.0) -> dict:
        g_k = self.lift(x_k, tau=tau)
        g_k1 = self.lift(x_k1, tau=tau)
        g_k1_pred = self.predict(g_k)
        x_k_recon = self.reconstruct(g_k)

        return {
            "pred_loss": F.mse_loss(g_k1_pred, g_k1),
            "recon_loss": F.mse_loss(x_k_recon, x_k),
            "reg_loss": torch.norm(self.K, p="fro") ** 2,
            "g_k": g_k,
            "g_k1": g_k1,
            "g_k1_pred": g_k1_pred,
            "x_k_recon": x_k_recon,
        }
