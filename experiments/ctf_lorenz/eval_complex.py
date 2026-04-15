"""Compare real vs complex EML-Koopman on CTF Lorenz (E1, E2).

Configurations:
  1. Real baseline         -- original grammar
  2. i+ix (original)       -- complex grammar, no fixes
  3. i+ix + child bias     -- Fix 1: bias child-node logit by +2.0
  4. i+ix + warm-start     -- Fix 2: pre-train real, then expand to complex
  5. i+ix + mixed dict     -- Fix 3: 8 real + 8 complex trees
  6. i+ix + slow anneal    -- Fix 4: 75% exploration phase, tau_start=3.0
  7. i+ix + all fixes      -- Fix 1+3+4 combined (bias + mixed + slow anneal)
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
from koopman_eml.training import train_koopman_eml, train_warmstart_complex
from experiments.ctf_lorenz.generate_data import generate_lorenz_trajectories


def _run_one(
    label: str,
    data: dict,
    model: KoopmanEML,
    train_fn,
    train_kwargs: dict,
    device: str,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print("=" * 70)

    X_k_np = data["X_k_train"]
    X_k1_np = data["X_k1_train"]
    rng = np.random.default_rng(0)
    max_train_pairs = 20_000
    if len(X_k_np) > max_train_pairs:
        idx = rng.choice(len(X_k_np), max_train_pairs, replace=False)
        X_k_np, X_k1_np = X_k_np[idx], X_k1_np[idx]

    X_k = torch.tensor(X_k_np, dtype=torch.float32)
    X_k1 = torch.tensor(X_k1_np, dtype=torch.float32)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  complex={model.use_complex}  "
          f"mixed={model._mixed}")

    t0 = time.time()
    train_fn(model, X_k, X_k1, device=device, verbose=True, **train_kwargs)
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

    depth2_count = sum(1 for f in formulas if f.count("eml(") >= 2)

    result = {
        "label": label,
        "E1": e1,
        "E2": e2,
        "rmse": basic["rmse"],
        "valid_prediction_steps": basic["valid_prediction_steps"],
        "n_params": n_params,
        "train_time_s": train_time,
        "depth2_trees": depth2_count,
        "formulas": formulas,
    }
    print(f"\n  E1 = {e1:.2f}   E2 = {e2:.2f}   RMSE = {basic['rmse']:.4f}   "
          f"valid_steps = {basic['valid_prediction_steps']}   "
          f"depth-2 trees = {depth2_count}/{len(formulas)}")
    return result


def eval_complex(
    n_observables: int = 16,
    tree_depth: int = 2,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    batch_size: int = 2048,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/complex",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("#" * 70)
    print("#  COMPLEX EML COMPARISON v2  --  With depth-collapse fixes")
    print("#" * 70)

    data = generate_lorenz_trajectories()
    std_train = dict(n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    all_results = {}

    # --- 1. Real baseline ---
    model = KoopmanEML(state_dim=3, n_observables=n_observables,
                       tree_depth=tree_depth, exp_order=10, ln_order=12)
    all_results["real_baseline"] = _run_one(
        "real_baseline", data, model, train_koopman_eml, std_train, device)

    # --- 2. i+ix original (no fixes) ---
    model = KoopmanEML(state_dim=3, n_observables=n_observables,
                       tree_depth=tree_depth, exp_order=10, ln_order=12,
                       use_complex=True, allow_imaginary_vars=True)
    all_results["ix_original"] = _run_one(
        "ix_original", data, model, train_koopman_eml, std_train, device)

    # --- 3. Fix 1: child logit bias ---
    model = KoopmanEML(state_dim=3, n_observables=n_observables,
                       tree_depth=tree_depth, exp_order=10, ln_order=12,
                       use_complex=True, allow_imaginary_vars=True,
                       child_logit_bias=2.0)
    all_results["ix_child_bias"] = _run_one(
        "ix_child_bias", data, model, train_koopman_eml, std_train, device)

    # --- 4. Fix 2: warm-start ---
    model = KoopmanEML(state_dim=3, n_observables=n_observables,
                       tree_depth=tree_depth, exp_order=10, ln_order=12,
                       use_complex=True, allow_imaginary_vars=True)
    warmstart_kwargs = dict(pretrain_epochs=800, complex_epochs=400,
                            lr=lr, batch_size=batch_size)
    all_results["ix_warmstart"] = _run_one(
        "ix_warmstart", data, model, train_warmstart_complex,
        warmstart_kwargs, device)

    # --- 5. Fix 3: mixed dictionary (8 real + 8 complex) ---
    model = KoopmanEML(state_dim=3, n_observables=n_observables,
                       tree_depth=tree_depth, exp_order=10, ln_order=12,
                       use_complex=True, allow_imaginary_vars=True,
                       n_complex_trees=n_observables // 2)
    all_results["ix_mixed_dict"] = _run_one(
        "ix_mixed_dict", data, model, train_koopman_eml, std_train, device)

    # --- 6. Fix 4: slow anneal ---
    model = KoopmanEML(state_dim=3, n_observables=n_observables,
                       tree_depth=tree_depth, exp_order=10, ln_order=12,
                       use_complex=True, allow_imaginary_vars=True)
    slow_train = dict(n_epochs=n_epochs, lr=lr, batch_size=batch_size,
                      tau_start=3.0, phase1_frac=0.75, phase2_frac=0.15)
    all_results["ix_slow_anneal"] = _run_one(
        "ix_slow_anneal", data, model, train_koopman_eml, slow_train, device)

    # --- 7. All fixes combined (bias + mixed + slow anneal) ---
    model = KoopmanEML(state_dim=3, n_observables=n_observables,
                       tree_depth=tree_depth, exp_order=10, ln_order=12,
                       use_complex=True, allow_imaginary_vars=True,
                       child_logit_bias=2.0,
                       n_complex_trees=n_observables // 2)
    all_results["ix_all_fixes"] = _run_one(
        "ix_all_fixes", data, model, train_koopman_eml, slow_train, device)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)
    print(f"  {'Config':<22} {'E1':>7} {'E2':>7} {'RMSE':>7} {'Steps':>6} "
          f"{'D2':>4} {'Params':>7} {'Time':>6}")
    print("  " + "-" * 72)
    for label, r in all_results.items():
        print(f"  {label:<22} {r['E1']:>7.2f} {r['E2']:>7.2f} "
              f"{r['rmse']:>7.4f} {r['valid_prediction_steps']:>6} "
              f"{r['depth2_trees']:>4} {r['n_params']:>7,} "
              f"{r['train_time_s']:>5.0f}s")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk != "formulas"}
        for k, v in all_results.items()
    }
    with open(out / "complex_comparison_v2.json", "w") as f:
        json.dump(serializable, f, indent=2)

    formulas_out = {k: v["formulas"] for k, v in all_results.items()}
    with open(out / "complex_formulas_v2.json", "w") as f:
        json.dump(formulas_out, f, indent=2)

    print(f"\n  Results saved to {out}")
    return all_results


if __name__ == "__main__":
    eval_complex()
