"""E11-E12: Parametric generalization evaluation.

E11 -- Interpolation: train on rho = {26, 30}, forecast at rho = 28.
E12 -- Extrapolation: train on rho = {26, 28}, forecast at rho = 32.

Tests whether the learned Koopman operator generalizes across the
bifurcation parameter space.  Because EML observables are closed-form
symbolic functions (exp/ln compositions), they can represent parameter-
independent geometric structure of the attractor, potentially giving
better generalization than purely numerical methods.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from koopman_eml import KoopmanEML
from koopman_eml.analysis import prediction_rollout
from koopman_eml.ctf import short_term_score
from koopman_eml.training import train_koopman_eml
from experiments.ctf_lorenz.generate_ctf_scenarios import generate_all_ctf_scenarios


def _train_model(
    X_k_np: np.ndarray,
    X_k1_np: np.ndarray,
    n_observables: int,
    tree_depth: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    max_train_pairs: int,
    device: str,
    label: str = "",
) -> tuple[KoopmanEML, float]:
    rng = np.random.default_rng(0)
    if len(X_k_np) > max_train_pairs:
        idx = rng.choice(len(X_k_np), max_train_pairs, replace=False)
        X_k_np, X_k1_np = X_k_np[idx], X_k1_np[idx]

    X_k = torch.tensor(X_k_np, dtype=torch.float32)
    X_k1 = torch.tensor(X_k1_np, dtype=torch.float32)
    print(f"  [{label}] Training on {len(X_k):,} pairs, device: {device}")

    model = KoopmanEML(
        state_dim=3, n_observables=n_observables, tree_depth=tree_depth,
        exp_order=10, ln_order=12,
    )

    t0 = time.time()
    train_koopman_eml(
        model, X_k, X_k1,
        n_epochs=n_epochs, lr=lr, device=device, batch_size=batch_size, verbose=True,
    )
    return model, time.time() - t0


def _forecast_and_score(
    model: KoopmanEML, train_seq: np.ndarray, test_seq: np.ndarray, device: str,
) -> tuple[float, np.ndarray]:
    x0 = torch.tensor(train_seq[-1], dtype=torch.float32)
    n_forecast = len(test_seq)
    traj = prediction_rollout(model, x0, n_forecast - 1, device=device)
    pred = traj[1:].numpy()
    truth = test_seq[:len(pred)]
    n_eval = min(len(pred), len(truth))
    pred, truth = pred[:n_eval], truth[:n_eval]
    return short_term_score(pred, truth), pred


def eval_e11_e12(
    n_observables: int = 16,
    tree_depth: int = 2,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    batch_size: int = 2048,
    max_train_pairs: int = 20_000,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/e11_e12",
    data: dict | None = None,
) -> dict[str, float]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E11-E12: Parametric Generalization Evaluation")
    print("=" * 70)

    if data is None:
        data = generate_all_ctf_scenarios()

    # --- E11: interpolation (train rho={26,30}, test rho=28) ---
    print("\n--- E11: parametric interpolation ---")
    model_interp, time_interp = _train_model(
        data["X8_k_train"], data["X8_k1_train"],
        n_observables, tree_depth, n_epochs, lr, batch_size, max_train_pairs, device,
        label="interp",
    )
    e11, pred_interp = _forecast_and_score(
        model_interp, data["X8train"], data["X8test"], device,
    )
    print(f"  E11 = {e11:.2f}")

    # --- E12: extrapolation (train rho={26,28}, test rho=32) ---
    print("\n--- E12: parametric extrapolation ---")
    model_extrap, time_extrap = _train_model(
        data["X9_k_train"], data["X9_k1_train"],
        n_observables, tree_depth, n_epochs, lr, batch_size, max_train_pairs, device,
        label="extrap",
    )
    e12, pred_extrap = _forecast_and_score(
        model_extrap, data["X9train"], data["X9test"], device,
    )
    print(f"  E12 = {e12:.2f}")

    n_params = sum(p.numel() for p in model_interp.parameters())
    results = {
        "E11": e11,
        "E12": e12,
        "n_params": n_params,
        "train_time_interp_s": time_interp,
        "train_time_extrap_s": time_extrap,
    }

    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    for k, v in results.items():
        print(f"  {k:<28} {v:>12.4f}" if isinstance(v, float) else f"  {k:<28} {v:>12}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model_interp.state_dict(), out / "model_interp.pt")
    torch.save(model_extrap.state_dict(), out / "model_extrap.pt")
    np.save(out / "pred_interp.npy", pred_interp)
    np.save(out / "pred_extrap.npy", pred_extrap)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == "__main__":
    eval_e11_e12()
