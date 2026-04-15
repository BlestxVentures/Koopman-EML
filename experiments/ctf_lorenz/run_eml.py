"""Run EML-Koopman on the CTF Lorenz dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from koopman_eml import KoopmanEML
from koopman_eml.analysis import (
    compute_metrics,
    extract_eml_formulas,
    koopman_eigendecomposition,
    prediction_rollout,
)
from koopman_eml.ctf import evaluate_ctf, short_term_score
from koopman_eml.training import train_koopman_eml
from experiments.ctf_lorenz.generate_data import generate_lorenz_trajectories


def run(
    n_observables: int = 16,
    tree_depth: int = 3,
    n_epochs: int = 2000,
    lr: float = 1e-3,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/eml",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("EML-Koopman on CTF Lorenz")
    print("=" * 70)

    # Generate data
    data = generate_lorenz_trajectories()
    X_k = torch.tensor(data["X_k_train"], dtype=torch.float32)
    X_k1 = torch.tensor(data["X_k1_train"], dtype=torch.float32)

    # Build and train model
    model = KoopmanEML(state_dim=3, n_observables=n_observables, tree_depth=tree_depth)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_koopman_eml(
        model, X_k, X_k1,
        n_epochs=n_epochs, lr=lr, device=device, verbose=True,
    )

    # Eigendecomposition
    eig = koopman_eigendecomposition(model)
    print("\nTop eigenvalues:")
    for i in range(min(8, len(eig["eigenvalues"]))):
        lam = eig["eigenvalues"][i]
        print(f"  lambda_{i} = {lam:.4f}  |lambda|={abs(lam):.4f}")

    # Snapped formulas
    formulas = extract_eml_formulas(model)
    print("\nDiscovered EML observables:")
    for i, f in enumerate(formulas):
        print(f"  g_{i}(x) = {f}")

    # Forecasting evaluation
    x0 = torch.tensor(data["X1train"][-1], dtype=torch.float32)
    n_forecast = len(data["X1test"])
    traj = prediction_rollout(model, x0, n_forecast - 1, device=device)
    pred = traj[1:].numpy()
    truth = data["X1test"]

    n_eval = min(len(pred), len(truth))
    pred, truth = pred[:n_eval], truth[:n_eval]

    metrics = compute_metrics(pred, truth)
    print(f"\nForecasting: RMSE={metrics['rmse']:.4f}, valid_steps={metrics['valid_prediction_steps']}")

    e1 = short_term_score(pred, truth)
    print(f"CTF E1 (short-term): {e1:.2f}")

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    np.save(out / "predictions.npy", pred)
    np.save(out / "eigenvalues.npy", eig["eigenvalues"])
    with open(out / "formulas.json", "w") as f:
        json.dump(formulas, f, indent=2)
    with open(out / "metrics.json", "w") as f:
        json.dump({**metrics, "E1": e1}, f, indent=2)
    with open(out / "history.json", "w") as f:
        json.dump(history, f)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    run()
