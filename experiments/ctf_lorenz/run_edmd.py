"""Run EDMD baselines on CTF Lorenz."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baselines.edmd import EDMDModel
from koopman_eml.ctf import short_term_score
from koopman_eml.analysis import compute_metrics
from experiments.ctf_lorenz.generate_data import generate_lorenz_trajectories


def run(dictionary: str = "poly", output_dir: str = "results/ctf_lorenz/edmd"):
    print("=" * 70)
    print(f"EDMD ({dictionary}) on CTF Lorenz")
    print("=" * 70)

    data = generate_lorenz_trajectories()
    X_k = data["X_k_train"]
    X_k1 = data["X_k1_train"]

    model = EDMDModel(dictionary=dictionary, poly_degree=4, n_rbf=200, rbf_sigma=2.0)
    model.fit(X_k, X_k1)

    x0 = data["X1train"][-1]
    n_forecast = len(data["X1test"])
    pred = model.predict(x0, n_forecast - 1)[1:]
    truth = data["X1test"]

    n_eval = min(len(pred), len(truth))
    pred, truth = pred[:n_eval], truth[:n_eval]

    metrics = compute_metrics(pred, truth)
    e1 = short_term_score(pred, truth)
    print(f"RMSE={metrics['rmse']:.4f}, valid_steps={metrics['valid_prediction_steps']}, E1={e1:.2f}")

    out = Path(f"{output_dir}_{dictionary}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "predictions.npy", pred)
    with open(out / "metrics.json", "w") as f:
        json.dump({**metrics, "E1": e1}, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    run("poly")
    run("rbf")
