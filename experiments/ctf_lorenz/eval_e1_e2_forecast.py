"""E1-E2: Clean forecasting evaluation.

E1 -- Short-term forecasting: how accurately does the Koopman model track
      the true trajectory before chaotic divergence?
E2 -- Long-term spectral:     does the model reproduce the correct power
      spectrum (attractor statistics) even after trajectories diverge?

Pipeline:
    1. Train EML-Koopman on clean consecutive-pair data (X_k, X_k+1).
    2. Roll out multi-step predictions from the last training point.
    3. Score with short_term_score (E1) and long_term_score (E2).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from koopman_eml import KoopmanEML
from koopman_eml.analysis import compute_metrics, extract_eml_formulas, prediction_rollout
from koopman_eml.ctf import long_term_score, short_term_score
from koopman_eml.training import train_koopman_eml
from experiments.ctf_lorenz.generate_ctf_scenarios import generate_all_ctf_scenarios


def eval_e1_e2(
    n_observables: int = 16,
    tree_depth: int = 2,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    batch_size: int = 2048,
    max_train_pairs: int = 20_000,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/e1_e2",
    data: dict | None = None,
) -> dict[str, float]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E1-E2: Clean Forecasting Evaluation")
    print("=" * 70)

    if data is None:
        data = generate_all_ctf_scenarios()

    X_k_np = data["X_k_train"]
    X_k1_np = data["X_k1_train"]

    rng = np.random.default_rng(0)
    if len(X_k_np) > max_train_pairs:
        idx = rng.choice(len(X_k_np), max_train_pairs, replace=False)
        X_k_np, X_k1_np = X_k_np[idx], X_k1_np[idx]

    X_k = torch.tensor(X_k_np, dtype=torch.float32)
    X_k1 = torch.tensor(X_k1_np, dtype=torch.float32)
    print(f"Training pairs: {len(X_k):,}, device: {device}")

    model = KoopmanEML(
        state_dim=3, n_observables=n_observables, tree_depth=tree_depth,
        exp_order=10, ln_order=12,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    t0 = time.time()
    train_koopman_eml(
        model, X_k, X_k1,
        n_epochs=n_epochs, lr=lr, device=device, batch_size=batch_size, verbose=True,
    )
    train_time = time.time() - t0

    formulas = extract_eml_formulas(model)
    print("\nSnapped observables:")
    for i, f in enumerate(formulas[:8]):
        print(f"  g_{i}(x) = {f}")

    x0 = torch.tensor(data["X1train"][-1], dtype=torch.float32)
    n_forecast = len(data["X1test"])
    traj = prediction_rollout(model, x0, n_forecast - 1, device=device)
    pred = traj[1:].numpy()
    truth = data["X1test"][:len(pred)]

    n_eval = min(len(pred), len(truth))
    pred, truth = pred[:n_eval], truth[:n_eval]

    basic = compute_metrics(pred, truth)
    e1 = short_term_score(pred, truth)
    e2 = long_term_score(pred, truth)

    results = {
        "E1": e1,
        "E2": e2,
        "rmse": basic["rmse"],
        "relative_error": basic["relative_error"],
        "valid_prediction_steps": basic["valid_prediction_steps"],
        "n_params": n_params,
        "train_time_s": train_time,
        "n_forecast_steps": n_eval,
    }

    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    for k, v in results.items():
        print(f"  {k:<28} {v:>12.4f}" if isinstance(v, float) else f"  {k:<28} {v:>12}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    np.save(out / "predictions.npy", pred)
    with open(out / "formulas.json", "w") as f:
        json.dump(formulas, f, indent=2)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == "__main__":
    eval_e1_e2()
