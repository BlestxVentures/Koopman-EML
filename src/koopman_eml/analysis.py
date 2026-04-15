"""
Koopman eigendecomposition, EML formula extraction, and evaluation utilities.
"""

from __future__ import annotations

import numpy as np
import torch

from koopman_eml.koopman_model import KoopmanEML


def koopman_eigendecomposition(model: KoopmanEML) -> dict:
    """Extract eigenvalues, eigenvectors, modes, frequencies, and growth rates."""
    K = model.K.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(K)

    C = model.C.weight.detach().cpu().numpy()
    modes = C @ eigenvectors

    ln_eig = np.log(eigenvalues.astype(complex) + 1e-30)
    frequencies = ln_eig.imag / (2 * np.pi)
    growth_rates = ln_eig.real

    order = np.argsort(-np.abs(eigenvalues))
    return {
        "eigenvalues": eigenvalues[order],
        "eigenvectors": eigenvectors[:, order],
        "modes": modes[:, order],
        "frequencies": frequencies[order],
        "growth_rates": growth_rates[order],
    }


def extract_eml_formulas(model: KoopmanEML) -> list[str]:
    """Read out the symbolic EML formula for each observable after snapping."""
    dict_mod = model.dictionary
    n_vars = dict_mod.n_vars
    var_names = [f"x{i}" for i in range(n_vars)]
    formulas: list[str] = []

    for tree_idx in range(dict_mod.n_trees):
        prev_exprs: list[str] | None = None

        for level in range(dict_mod.depth):
            logits = dict_mod.level_logits[level]
            n_nodes = dict_mod.nodes_per_level[level]
            current_exprs: list[str] = []

            for node_idx in range(n_nodes):
                left_idx = logits[tree_idx, node_idx, :, 0].argmax().item()
                right_idx = logits[tree_idx, node_idx, :, 1].argmax().item()

                def _resolve(idx: int, node_i: int) -> str:
                    if idx == 0:
                        return "1"
                    if idx <= n_vars:
                        return var_names[idx - 1]
                    if prev_exprs is not None:
                        child_i = 2 * node_i + (0 if idx == n_vars + 1 else 1)
                        if child_i < len(prev_exprs):
                            return prev_exprs[child_i]
                    return "?"

                left_expr = _resolve(left_idx, node_idx)
                right_expr = _resolve(right_idx, node_idx)

                if left_expr == "?" and prev_exprs is not None and 2 * node_idx < len(prev_exprs):
                    left_expr = prev_exprs[2 * node_idx]
                if right_expr == "?" and prev_exprs is not None and 2 * node_idx + 1 < len(prev_exprs):
                    right_expr = prev_exprs[2 * node_idx + 1]

                current_exprs.append(f"eml({left_expr}, {right_expr})")

            prev_exprs = current_exprs

        formulas.append(prev_exprs[0])

    return formulas


def prediction_rollout(
    model: KoopmanEML,
    x0: torch.Tensor,
    n_steps: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Multi-step rollout in Koopman space.

    Args:
        x0: [state_dim] initial state
        n_steps: number of steps to predict

    Returns:
        Tensor of shape [n_steps + 1, state_dim] -- trajectory starting at x0.
    """
    model.eval()
    model = model.to(device)
    x0 = x0.to(device)

    trajectory = [x0.unsqueeze(0)]
    with torch.no_grad():
        g = model.lift(x0.unsqueeze(0), tau=0.01)
        for _ in range(n_steps):
            g = model.predict(g)
            x_hat = model.reconstruct(g)
            trajectory.append(x_hat)

    return torch.cat(trajectory, dim=0).cpu()


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Compute standard evaluation metrics between predicted and true trajectories."""
    rmse = float(np.sqrt(np.mean((predictions - ground_truth) ** 2)))
    rel_err = float(np.linalg.norm(predictions - ground_truth) / (np.linalg.norm(ground_truth) + 1e-30))

    per_step_err = np.sqrt(np.mean((predictions - ground_truth) ** 2, axis=-1))
    valid_steps = int(np.sum(per_step_err < 0.5 * np.std(ground_truth)))

    return {
        "rmse": rmse,
        "relative_error": rel_err,
        "valid_prediction_steps": valid_steps,
    }
