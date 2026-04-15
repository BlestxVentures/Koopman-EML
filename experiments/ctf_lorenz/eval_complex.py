"""Compare real vs complex EML-Koopman on CTF Lorenz (E1, E2).

Three configurations:
  1. Real baseline        -- use_complex=False  (original)
  2. i-only complex       -- use_complex=True, allow_imaginary_vars=False
  3. i+ix complex         -- use_complex=True, allow_imaginary_vars=True

Each is trained from scratch with the same hyperparameters and data,
then evaluated on short-term (E1) and long-term (E2) CTF metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from koopman_eml import KoopmanEML
from koopman_eml.analysis import (
    compute_metrics,
    extract_eml_formulas,
    prediction_rollout,
)
from koopman_eml.ctf import long_term_score, short_term_score
from koopman_eml.training import train_koopman_eml
from experiments.ctf_lorenz.generate_data import generate_lorenz_trajectories


CONFIGS = {
    "real_baseline": dict(use_complex=False, allow_imaginary_vars=False),
    "complex_i_only": dict(use_complex=True, allow_imaginary_vars=False),
    "complex_i_plus_ix": dict(use_complex=True, allow_imaginary_vars=True),
}


def _run_one(
    label: str,
    data: dict,
    use_complex: bool,
    allow_imaginary_vars: bool,
    n_observables: int,
    tree_depth: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    max_train_pairs: int,
    device: str,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print("=" * 70)

    X_k_np = data["X_k_train"]
    X_k1_np = data["X_k1_train"]
    rng = np.random.default_rng(0)
    if len(X_k_np) > max_train_pairs:
        idx = rng.choice(len(X_k_np), max_train_pairs, replace=False)
        X_k_np, X_k1_np = X_k_np[idx], X_k1_np[idx]

    X_k = torch.tensor(X_k_np, dtype=torch.float32)
    X_k1 = torch.tensor(X_k1_np, dtype=torch.float32)

    model = KoopmanEML(
        state_dim=3,
        n_observables=n_observables,
        tree_depth=tree_depth,
        exp_order=10,
        ln_order=12,
        use_complex=use_complex,
        allow_imaginary_vars=allow_imaginary_vars,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  complex={use_complex}  imag_vars={allow_imaginary_vars}")

    t0 = time.time()
    train_koopman_eml(
        model, X_k, X_k1,
        n_epochs=n_epochs, lr=lr, device=device,
        batch_size=batch_size, verbose=True,
    )
    train_time = time.time() - t0

    formulas = extract_eml_formulas(model)
    print("\n  Snapped observables:")
    for i, f in enumerate(formulas[:8]):
        print(f"    g_{i}(x) = {f}")

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

    result = {
        "label": label,
        "E1": e1,
        "E2": e2,
        "rmse": basic["rmse"],
        "valid_prediction_steps": basic["valid_prediction_steps"],
        "n_params": n_params,
        "train_time_s": train_time,
        "use_complex": use_complex,
        "allow_imaginary_vars": allow_imaginary_vars,
        "formulas": formulas,
    }
    print(f"\n  E1 = {e1:.2f}   E2 = {e2:.2f}   RMSE = {basic['rmse']:.4f}   "
          f"valid_steps = {basic['valid_prediction_steps']}")
    return result


def eval_complex(
    n_observables: int = 16,
    tree_depth: int = 2,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    batch_size: int = 2048,
    max_train_pairs: int = 20_000,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/complex",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("#" * 70)
    print("#  COMPLEX EML COMPARISON  --  Real vs i-only vs i+ix")
    print("#" * 70)

    data = generate_lorenz_trajectories()

    all_results = {}
    for label, cfg in CONFIGS.items():
        all_results[label] = _run_one(
            label=label, data=data, **cfg,
            n_observables=n_observables, tree_depth=tree_depth,
            n_epochs=n_epochs, lr=lr, batch_size=batch_size,
            max_train_pairs=max_train_pairs, device=device,
        )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<25} {'E1':>8} {'E2':>8} {'RMSE':>8} {'Steps':>7} {'Params':>8} {'Time':>7}")
    print("  " + "-" * 75)
    for label, r in all_results.items():
        print(f"  {label:<25} {r['E1']:>8.2f} {r['E2']:>8.2f} "
              f"{r['rmse']:>8.4f} {r['valid_prediction_steps']:>7} "
              f"{r['n_params']:>8,} {r['train_time_s']:>6.0f}s")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk != "formulas"}
        for k, v in all_results.items()
    }
    with open(out / "complex_comparison.json", "w") as f:
        json.dump(serializable, f, indent=2)

    formulas_out = {k: v["formulas"] for k, v in all_results.items()}
    with open(out / "complex_formulas.json", "w") as f:
        json.dump(formulas_out, f, indent=2)

    print(f"\n  Results saved to {out}")
    return all_results


if __name__ == "__main__":
    eval_complex()
