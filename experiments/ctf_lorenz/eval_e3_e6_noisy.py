"""E3-E6: Noisy data evaluation.

E3 -- Short-term reconstruction from medium noise (SNR ≈ 20 dB).
E4 -- Long-term forecasting from medium-noise observations.
E5 -- Short-term reconstruction from high noise (SNR ≈ 10 dB).
E6 -- Long-term forecasting from high-noise observations.

Pipeline for each noise level:
    1. Train EML-Koopman on *clean* consecutive-pair data to learn the
       Koopman operator and low-dimensional observable manifold.
    2. Reconstruction (E3/E5): lift the noisy training window into
       observable space, then decode — the Koopman lifting acts as a
       learned denoiser because the observable dictionary spans a
       lower-dimensional manifold than the ambient noise.
    3. Forecast (E4/E6): starting from the last reconstructed state,
       roll out multi-step predictions and score against clean truth.
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


def _reconstruct(model: KoopmanEML, noisy: np.ndarray, device: str) -> np.ndarray:
    """Denoise via lift -> reconstruct."""
    model.eval()
    x_noisy = torch.tensor(noisy, dtype=torch.float32).to(device)
    with torch.no_grad():
        g = model.lift(x_noisy, tau=0.01)
        x_clean = model.reconstruct(g)
    return x_clean.cpu().numpy()


def _forecast(model: KoopmanEML, x0: np.ndarray, n_steps: int, device: str) -> np.ndarray:
    x0_t = torch.tensor(x0, dtype=torch.float32)
    traj = prediction_rollout(model, x0_t, n_steps, device=device)
    return traj[1:].numpy()


def _train_clean_model(
    data: dict,
    n_observables: int,
    tree_depth: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    max_train_pairs: int,
    device: str,
) -> tuple[KoopmanEML, float]:
    X_k_np = data["X_k_train"]
    X_k1_np = data["X_k1_train"]

    rng = np.random.default_rng(0)
    if len(X_k_np) > max_train_pairs:
        idx = rng.choice(len(X_k_np), max_train_pairs, replace=False)
        X_k_np, X_k1_np = X_k_np[idx], X_k1_np[idx]

    X_k = torch.tensor(X_k_np, dtype=torch.float32)
    X_k1 = torch.tensor(X_k1_np, dtype=torch.float32)

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


def eval_e3_e6(
    n_observables: int = 16,
    tree_depth: int = 2,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    batch_size: int = 2048,
    max_train_pairs: int = 20_000,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/e3_e6",
    data: dict | None = None,
) -> dict[str, float]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E3-E6: Noisy Data Evaluation")
    print("=" * 70)

    if data is None:
        data = generate_all_ctf_scenarios()

    model, train_time = _train_clean_model(
        data, n_observables, tree_depth, n_epochs, lr, batch_size, max_train_pairs, device,
    )
    n_params = sum(p.numel() for p in model.parameters())

    # --- E3: medium-noise reconstruction ---
    print("\n--- E3: medium-noise reconstruction ---")
    X2_noisy = data["X2train"]
    X2_clean_truth = data["X2test"]
    X2_recon = _reconstruct(model, X2_noisy, device)
    n_eval = min(len(X2_recon), len(X2_clean_truth))
    e3 = short_term_score(X2_recon[:n_eval], X2_clean_truth[:n_eval])
    print(f"  E3 = {e3:.2f}")

    # --- E4: medium-noise long-term forecast ---
    print("--- E4: medium-noise long-term forecast ---")
    X3_noisy_train = data["X3train"]
    last_recon = _reconstruct(model, X3_noisy_train[-1:], device)[0]
    n_forecast = len(data["X3test"])
    X3_pred = _forecast(model, last_recon, n_forecast - 1, device)
    X3_truth = data["X3test"][:len(X3_pred)]
    e4 = long_term_score(X3_pred, X3_truth)
    print(f"  E4 = {e4:.2f}")

    # --- E5: high-noise reconstruction ---
    print("--- E5: high-noise reconstruction ---")
    X4_noisy = data["X4train"]
    X4_clean_truth = data["X4test"]
    X4_recon = _reconstruct(model, X4_noisy, device)
    n_eval = min(len(X4_recon), len(X4_clean_truth))
    e5 = short_term_score(X4_recon[:n_eval], X4_clean_truth[:n_eval])
    print(f"  E5 = {e5:.2f}")

    # --- E6: high-noise long-term forecast ---
    print("--- E6: high-noise long-term forecast ---")
    X5_noisy_train = data["X5train"]
    last_recon = _reconstruct(model, X5_noisy_train[-1:], device)[0]
    n_forecast = len(data["X5test"])
    X5_pred = _forecast(model, last_recon, n_forecast - 1, device)
    X5_truth = data["X5test"][:len(X5_pred)]
    e6 = long_term_score(X5_pred, X5_truth)
    print(f"  E6 = {e6:.2f}")

    results = {
        "E3": e3,
        "E4": e4,
        "E5": e5,
        "E6": e6,
        "n_params": n_params,
        "train_time_s": train_time,
    }

    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    for k, v in results.items():
        print(f"  {k:<28} {v:>12.4f}" if isinstance(v, float) else f"  {k:<28} {v:>12}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == "__main__":
    eval_e3_e6()
