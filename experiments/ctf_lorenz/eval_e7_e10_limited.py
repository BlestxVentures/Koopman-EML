"""E7-E10: Limited data evaluation.

E7  -- Short-term forecasting trained on 10% clean data.
E8  -- Long-term spectral trained on 10% clean data.
E9  -- Short-term forecasting trained on 10% noisy data.
E10 -- Long-term spectral trained on 10% noisy data.

Tests how well EML-Koopman generalizes when the training budget is
severely reduced.  The symbolic observable dictionary (720 params at
depth 2) is a strong inductive bias in the low-data regime compared
to neural baselines with 10K-100K+ parameters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from koopman_eml import KoopmanEML
from koopman_eml.analysis import prediction_rollout
from koopman_eml.ctf import long_term_score, short_term_score
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
    device: str,
    label: str = "",
) -> tuple[KoopmanEML, float]:
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
    model: KoopmanEML, data: dict, train_key: str, test_key: str, device: str,
) -> tuple[float, float, np.ndarray]:
    x0 = torch.tensor(data[train_key][-1], dtype=torch.float32)
    n_forecast = len(data[test_key])
    traj = prediction_rollout(model, x0, n_forecast - 1, device=device)
    pred = traj[1:].numpy()
    truth = data[test_key][:len(pred)]
    n_eval = min(len(pred), len(truth))
    pred, truth = pred[:n_eval], truth[:n_eval]
    return (
        short_term_score(pred, truth),
        long_term_score(pred, truth),
        pred,
    )


def eval_e7_e10(
    n_observables: int = 16,
    tree_depth: int = 2,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    batch_size: int = 2048,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/e7_e10",
    data: dict | None = None,
) -> dict[str, float]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E7-E10: Limited Data Evaluation")
    print("=" * 70)

    if data is None:
        data = generate_all_ctf_scenarios()

    # --- E7/E8: limited clean data ---
    print("\n--- E7/E8: limited clean data ---")
    model_clean, time_clean = _train_model(
        data["X6_k_train"], data["X6_k1_train"],
        n_observables, tree_depth, n_epochs, lr, batch_size, device,
        label="limited-clean",
    )
    e7, e8, pred_clean = _forecast_and_score(
        model_clean, data, "X6train", "X6test", device,
    )
    print(f"  E7 (short-term) = {e7:.2f}")
    print(f"  E8 (long-term)  = {e8:.2f}")

    # --- E9/E10: limited noisy data ---
    print("\n--- E9/E10: limited noisy data ---")
    model_noisy, time_noisy = _train_model(
        data["X7_k_train"], data["X7_k1_train"],
        n_observables, tree_depth, n_epochs, lr, batch_size, device,
        label="limited-noisy",
    )
    e9, e10, pred_noisy = _forecast_and_score(
        model_noisy, data, "X7train", "X7test", device,
    )
    print(f"  E9  (short-term) = {e9:.2f}")
    print(f"  E10 (long-term)  = {e10:.2f}")

    n_params = sum(p.numel() for p in model_clean.parameters())
    results = {
        "E7": e7,
        "E8": e8,
        "E9": e9,
        "E10": e10,
        "n_params": n_params,
        "n_train_clean": len(data["X6_k_train"]),
        "n_train_noisy": len(data["X7_k_train"]),
        "train_time_clean_s": time_clean,
        "train_time_noisy_s": time_noisy,
    }

    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    for k, v in results.items():
        print(f"  {k:<28} {v:>12.4f}" if isinstance(v, float) else f"  {k:<28} {v:>12}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model_clean.state_dict(), out / "model_limited_clean.pt")
    torch.save(model_noisy.state_dict(), out / "model_limited_noisy.pt")
    np.save(out / "pred_clean.npy", pred_clean)
    np.save(out / "pred_noisy.npy", pred_noisy)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == "__main__":
    eval_e7_e10()
