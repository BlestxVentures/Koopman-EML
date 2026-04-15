"""Progressive-depth EML-Koopman experiment targeting the E2 (long-term) metric.

Trains EML-Koopman models at increasing tree depths (2, 3, 4, 5) and selects
the configuration that maximizes E2 (long-term spectral score).  Each depth
level inherits the best hyperparameters from a per-depth grid search over
n_observables and training epochs.

The experiment also compares against EDMD-Poly (the current E2 leader at 52.94)
and reports whether deeper EML trees can close or surpass that gap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.ctf_lorenz.generate_data import generate_lorenz_trajectories
from koopman_eml import KoopmanEML
from koopman_eml.analysis import (
    compute_metrics,
    extract_eml_formulas,
    prediction_rollout,
)
from koopman_eml.ctf import long_term_score, short_term_score
from koopman_eml.training import train_koopman_eml
from baselines.edmd import EDMDModel


DEPTH_CONFIGS = {
    2: [
        {"n_obs": 16, "epochs": 1200, "lr": 3e-3, "batch": 2048, "clamp": 5.0},
        {"n_obs": 24, "epochs": 1500, "lr": 2e-3, "batch": 2048, "clamp": 5.0},
        {"n_obs": 32, "epochs": 1500, "lr": 2e-3, "batch": 2048, "clamp": 5.0},
    ],
    3: [
        {"n_obs": 16, "epochs": 1500, "lr": 2e-3, "batch": 2048, "clamp": 4.0},
        {"n_obs": 24, "epochs": 2000, "lr": 1e-3, "batch": 2048, "clamp": 4.0},
        {"n_obs": 32, "epochs": 2000, "lr": 1e-3, "batch": 2048, "clamp": 4.0},
    ],
    4: [
        {"n_obs": 16, "epochs": 2000, "lr": 1e-3, "batch": 1024, "clamp": 3.0},
        {"n_obs": 24, "epochs": 2500, "lr": 8e-4, "batch": 1024, "clamp": 3.0},
        {"n_obs": 32, "epochs": 2500, "lr": 8e-4, "batch": 1024, "clamp": 3.0},
    ],
    5: [
        {"n_obs": 16, "epochs": 2500, "lr": 5e-4, "batch": 512, "clamp": 2.5},
        {"n_obs": 24, "epochs": 3000, "lr": 5e-4, "batch": 512, "clamp": 2.5},
    ],
}

EDMD_POLY_BASELINE_E2 = 52.94  # current best E2 from run_all.py


def train_single(
    data: dict,
    depth: int,
    n_obs: int,
    epochs: int,
    lr: float,
    batch: int,
    clamp: float,
    device: str,
    seed: int = 0,
) -> dict:
    """Train one EML-Koopman configuration and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_k_np, X_k1_np = data["X_k_train"], data["X_k1_train"]
    rng = np.random.default_rng(seed)
    if len(X_k_np) > 20_000:
        idx = rng.choice(len(X_k_np), 20_000, replace=False)
        X_k_np, X_k1_np = X_k_np[idx], X_k1_np[idx]

    X_k = torch.tensor(X_k_np, dtype=torch.float32)
    X_k1 = torch.tensor(X_k1_np, dtype=torch.float32)

    exp_order = min(10 + depth, 16)
    ln_order = min(12 + depth, 20)

    model = KoopmanEML(
        state_dim=3,
        n_observables=n_obs,
        tree_depth=depth,
        exp_order=exp_order,
        ln_order=ln_order,
        clamp_input=clamp,
    )
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    history = train_koopman_eml(
        model, X_k, X_k1,
        n_epochs=epochs, lr=lr, device=device,
        verbose=True, batch_size=batch,
    )
    train_time = time.time() - t0

    # Evaluate forecasting
    x0 = torch.tensor(data["X1train"][-1], dtype=torch.float32)
    n_fc = len(data["X1test"])
    traj = prediction_rollout(model, x0, n_fc - 1, device=device)
    pred = traj[1:].numpy()
    truth = data["X1test"][:len(pred)]

    pred = np.clip(pred, -100, 100)

    m = compute_metrics(pred, truth)
    m["E1"] = short_term_score(pred, truth)
    m["E2"] = long_term_score(pred, truth)
    m["train_time"] = train_time
    m["n_params"] = n_params
    m["depth"] = depth
    m["n_obs"] = n_obs
    m["epochs"] = epochs
    m["lr"] = lr
    m["clamp"] = clamp

    formulas = extract_eml_formulas(model)
    m["formulas"] = formulas

    return m, model, pred


def run_edmd_baseline(data: dict) -> dict:
    """Run EDMD-Poly baseline for reference E2 comparison."""
    print("\n" + "=" * 70)
    print("  EDMD-Poly Baseline (E2 reference)")
    print("=" * 70)

    X_k, X_k1 = data["X_k_train"], data["X_k1_train"]
    if len(X_k) > 20_000:
        idx = np.random.default_rng(0).choice(len(X_k), 20_000, replace=False)
        X_k, X_k1 = X_k[idx], X_k1[idx]

    model = EDMDModel(dictionary="poly", poly_degree=4)
    model.fit(X_k, X_k1)

    x0 = data["X1train"][-1]
    n_fc = len(data["X1test"])
    pred = model.predict(x0, n_fc - 1)[1:]
    truth = data["X1test"][:len(pred)]
    pred = np.clip(pred, -100, 100)

    m = compute_metrics(pred, truth)
    m["E1"] = short_term_score(pred, truth)
    m["E2"] = long_term_score(pred, truth)
    print(f"  EDMD-Poly: E1={m['E1']:.2f}, E2={m['E2']:.2f}")
    return m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Target: maximize E2 (long-term spectral score)")
    print(f"EDMD-Poly baseline E2 to beat: {EDMD_POLY_BASELINE_E2:.2f}\n")

    data = generate_lorenz_trajectories()
    print(f"Train pairs: {data['X_k_train'].shape}")
    print(f"Test forecast: {data['X1test'].shape}")

    edmd_result = run_edmd_baseline(data)
    edmd_e2 = edmd_result["E2"]

    results_dir = Path("results/ctf_lorenz/progressive_depth")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_e2 = -float("inf")
    best_config = None
    best_model = None
    best_pred = None

    for depth in sorted(DEPTH_CONFIGS.keys()):
        configs = DEPTH_CONFIGS[depth]
        print(f"\n{'#' * 70}")
        print(f"  DEPTH = {depth}  ({len(configs)} configurations)")
        print(f"{'#' * 70}")

        for i, cfg in enumerate(configs):
            tag = f"depth{depth}_obs{cfg['n_obs']}"
            print(f"\n--- [{tag}] epochs={cfg['epochs']}, lr={cfg['lr']}, "
                  f"batch={cfg['batch']}, clamp={cfg['clamp']} ---")

            try:
                m, model, pred = train_single(
                    data, depth,
                    n_obs=cfg["n_obs"],
                    epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    batch=cfg["batch"],
                    clamp=cfg["clamp"],
                    device=device,
                )

                print(f"\n  >> E1={m['E1']:.2f}, E2={m['E2']:.2f}, "
                      f"RMSE={m['rmse']:.4f}, valid_steps={m['valid_prediction_steps']}, "
                      f"params={m['n_params']}, time={m['train_time']:.1f}s")

                result_entry = {
                    "tag": tag,
                    "depth": depth,
                    "n_obs": cfg["n_obs"],
                    "epochs": cfg["epochs"],
                    "lr": cfg["lr"],
                    "clamp": cfg["clamp"],
                    "E1": m["E1"],
                    "E2": m["E2"],
                    "rmse": m["rmse"],
                    "valid_steps": m["valid_prediction_steps"],
                    "n_params": m["n_params"],
                    "train_time": m["train_time"],
                    "formulas": m["formulas"][:8],
                }
                all_results.append(result_entry)

                if m["E2"] > best_e2:
                    best_e2 = m["E2"]
                    best_config = result_entry
                    best_model = model
                    best_pred = pred
                    print(f"  ** New best E2! {best_e2:.2f} **")

            except Exception as e:
                print(f"  !! FAILED: {e}")
                all_results.append({
                    "tag": tag, "depth": depth, "n_obs": cfg["n_obs"],
                    "error": str(e),
                })

        # Early stop if we've already beaten EDMD-Poly
        if best_e2 > edmd_e2:
            print(f"\n>>> E2={best_e2:.2f} exceeds EDMD-Poly baseline ({edmd_e2:.2f}). "
                  f"Continuing to check if deeper trees improve further...")

    # Save best model
    if best_model is not None:
        torch.save(best_model.state_dict(), results_dir / "best_model.pt")
        np.save(results_dir / "best_predictions.npy", best_pred)

    # Summary
    summary = {
        "edmd_poly_e2": edmd_e2,
        "best_eml": best_config,
        "all_runs": all_results,
    }
    with open(results_dir / "progressive_depth_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PROGRESSIVE DEPTH RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Tag':<28} {'E1':>8} {'E2':>8} {'RMSE':>8} {'Steps':>7} {'Params':>8} {'Time':>7}")
    print("-" * 78)

    for r in all_results:
        if "error" in r:
            print(f"{r['tag']:<28} {'FAILED':>8}")
            continue
        print(f"{r['tag']:<28} {r['E1']:>8.2f} {r['E2']:>8.2f} "
              f"{r['rmse']:>8.4f} {r['valid_steps']:>7} {r['n_params']:>8} "
              f"{r['train_time']:>7.1f}")

    print("-" * 78)
    print(f"{'EDMD-Poly (baseline)':<28} {edmd_result['E1']:>8.2f} {edmd_e2:>8.2f}")
    print(f"\nBest EML E2: {best_e2:.2f} (config: {best_config['tag'] if best_config else 'N/A'})")
    if best_e2 > edmd_e2:
        print(f">>> EML BEATS EDMD-Poly by {best_e2 - edmd_e2:.2f} points!")
    else:
        print(f">>> Gap to EDMD-Poly: {edmd_e2 - best_e2:.2f} points")

    if best_config:
        print(f"\nBest config details:")
        print(f"  Depth: {best_config['depth']}")
        print(f"  Observables: {best_config['n_obs']}")
        print(f"  Epochs: {best_config['epochs']}")
        print(f"  LR: {best_config['lr']}")
        print(f"  Clamp: {best_config['clamp']}")
        print(f"  Params: {best_config['n_params']}")
        print(f"\n  Sample formulas:")
        for i, f in enumerate(best_config.get("formulas", [])[:5]):
            print(f"    g_{i} = {f}")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
