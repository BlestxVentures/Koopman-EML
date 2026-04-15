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
    """Koopman operator model whose observable dictionary is a bank of EML trees.

    When ``use_complex=True``, the imaginary unit *i* is added to the
    routing candidates, observables are complex-valued, K is a complex
    matrix, and the decoder C maps complex observables back to real
    state space via Re(C @ g).
    """

    def __init__(
        self,
        state_dim: int,
        n_observables: int = 32,
        tree_depth: int = 3,
        exp_order: int = 12,
        ln_order: int = 16,
        use_complex: bool = False,
        allow_imaginary_vars: bool = False,
        clamp_input: float = 5.0,
        child_logit_bias: float = 0.0,
        n_complex_trees: int | None = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_obs = n_observables
        self.exp_order = exp_order
        self.ln_order = ln_order
        self.use_complex = use_complex
        self.allow_imaginary_vars = allow_imaginary_vars
        self.clamp_input = clamp_input

        self._mixed = use_complex and n_complex_trees is not None and n_complex_trees < n_observables
        if self._mixed:
            n_real = n_observables - n_complex_trees
            self.n_real_trees = n_real
            self.dict_real = EMLTreeVectorized(
                n_trees=n_real, depth=tree_depth, n_vars=state_dim,
                use_complex=False, allow_imaginary_vars=False,
                child_logit_bias=child_logit_bias,
            )
            self.dict_complex = EMLTreeVectorized(
                n_trees=n_complex_trees, depth=tree_depth, n_vars=state_dim,
                use_complex=True, allow_imaginary_vars=allow_imaginary_vars,
                child_logit_bias=child_logit_bias,
            )
            self.dictionary = self.dict_complex
        else:
            self.n_real_trees = 0
            self.dictionary = EMLTreeVectorized(
                n_trees=n_observables, depth=tree_depth, n_vars=state_dim,
                use_complex=use_complex, allow_imaginary_vars=allow_imaginary_vars,
                child_logit_bias=child_logit_bias,
            )

        if use_complex:
            K_init = (
                torch.eye(n_observables, dtype=torch.cfloat)
                + torch.randn(n_observables, n_observables) * 0.01
                + 1j * torch.randn(n_observables, n_observables) * 0.01
            )
            self.K = nn.Parameter(K_init)
            self.C_re = nn.Linear(n_observables, state_dim, bias=False)
            self.C_im = nn.Linear(n_observables, state_dim, bias=False)
            nn.init.zeros_(self.C_im.weight)
        else:
            self.K = nn.Parameter(
                torch.eye(n_observables) + torch.randn(n_observables, n_observables) * 0.01
            )
            self.C = nn.Linear(n_observables, state_dim, bias=False)

    def lift(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Lift state to observable space: [B, state_dim] -> [B, n_obs]."""
        if self._mixed:
            g_real = self.dict_real(
                x, tau=tau, exp_order=self.exp_order, ln_order=self.ln_order,
                use_complex=False, clamp_input=self.clamp_input,
            )
            g_cplx = self.dict_complex(
                x, tau=tau, exp_order=self.exp_order, ln_order=self.ln_order,
                use_complex=True, clamp_input=self.clamp_input,
            )
            return torch.cat([g_real.to(g_cplx.dtype), g_cplx], dim=-1)
        return self.dictionary(
            x, tau=tau, exp_order=self.exp_order, ln_order=self.ln_order,
            use_complex=self.use_complex, clamp_input=self.clamp_input,
        )

    def predict(self, g: torch.Tensor) -> torch.Tensor:
        """One-step prediction in observable space."""
        return g @ self.K.T

    def reconstruct(self, g: torch.Tensor) -> torch.Tensor:
        """Decode from observable space back to real state space."""
        if self.use_complex:
            return self.C_re(g.real) + self.C_im(g.imag)
        return self.C(g)

    def forward(self, x_k: torch.Tensor, x_k1: torch.Tensor, tau: float = 1.0) -> dict:
        g_k = self.lift(x_k, tau=tau)
        g_k1 = self.lift(x_k1, tau=tau)
        g_k1_pred = self.predict(g_k)
        x_k_recon = self.reconstruct(g_k)

        if self.use_complex:
            pred_loss = torch.mean(torch.abs(g_k1_pred - g_k1) ** 2)
        else:
            pred_loss = F.mse_loss(g_k1_pred, g_k1)

        return {
            "pred_loss": pred_loss,
            "recon_loss": F.mse_loss(x_k_recon, x_k),
            "reg_loss": torch.real(torch.norm(self.K, p="fro") ** 2),
            "g_k": g_k,
            "g_k1": g_k1,
            "g_k1_pred": g_k1_pred,
            "x_k_recon": x_k_recon,
        }
