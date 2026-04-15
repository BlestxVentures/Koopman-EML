"""Unified E1-E12 CTF evaluation runner for EML-Koopman.

Generates all data scenarios once, then runs each metric group and
produces a combined scorecard, radar plot, and JSON summary.

Usage:
    python -m experiments.ctf_lorenz.eval_full_ctf
    python -m experiments.ctf_lorenz.eval_full_ctf --device cpu --output results/ctf_full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from koopman_eml.ctf import evaluate_ctf, radar_plot
from experiments.ctf_lorenz.generate_ctf_scenarios import generate_all_ctf_scenarios
from experiments.ctf_lorenz.eval_e1_e2_forecast import eval_e1_e2
from experiments.ctf_lorenz.eval_e3_e6_noisy import eval_e3_e6
from experiments.ctf_lorenz.eval_e7_e10_limited import eval_e7_e10
from experiments.ctf_lorenz.eval_e11_e12_parametric import eval_e11_e12


CTF_LEADERBOARD = {
    "LSTM": 64.54,
    "DeepONet": 57.80,
    "Reservoir": 54.87,
    "KAN": 47.28,
    "ODE-LSTM": 41.67,
    "SINDy": -3.00,
    "PyKoopman": -20.11,
}


def run_full_ctf(
    n_observables: int = 16,
    tree_depth: int = 2,
    n_epochs: int = 1200,
    lr: float = 3e-3,
    batch_size: int = 2048,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/full_ctf",
) -> dict[str, float]:
    wall_start = time.time()

    print("#" * 70)
    print("#  FULL CTF E1-E12 EVALUATION  —  EML-Koopman")
    print("#" * 70)

    print("\n[1/5] Generating all CTF data scenarios...")
    data = generate_all_ctf_scenarios()
    print("  Done.\n")

    model_kwargs = dict(
        n_observables=n_observables,
        tree_depth=tree_depth,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    # --- Run each metric group ---
    print("[2/5] E1-E2: clean forecasting")
    r12 = eval_e1_e2(**model_kwargs, data=data, output_dir=f"{output_dir}/e1_e2")

    print("\n[3/5] E3-E6: noisy data")
    r36 = eval_e3_e6(**model_kwargs, data=data, output_dir=f"{output_dir}/e3_e6")

    print("\n[4/5] E7-E10: limited data")
    r710 = eval_e7_e10(**model_kwargs, data=data, output_dir=f"{output_dir}/e7_e10")

    print("\n[5/5] E11-E12: parametric generalization")
    r1112 = eval_e11_e12(**model_kwargs, data=data, output_dir=f"{output_dir}/e11_e12")

    # --- Combine scores ---
    scores = {
        "E1": r12["E1"],
        "E2": r12["E2"],
        "E3": r36["E3"],
        "E4": r36["E4"],
        "E5": r36["E5"],
        "E6": r36["E6"],
        "E7": r710["E7"],
        "E8": r710["E8"],
        "E9": r710["E9"],
        "E10": r710["E10"],
        "E11": r1112["E11"],
        "E12": r1112["E12"],
    }
    scores["avg"] = float(np.mean([scores[f"E{i}"] for i in range(1, 13)]))
    wall_time = time.time() - wall_start

    # --- Scorecard ---
    print("\n" + "=" * 70)
    print("  FULL CTF SCORECARD  —  EML-Koopman")
    print("=" * 70)
    print(f"\n  {'Metric':<8} {'Category':<35} {'Score':>8}")
    print("  " + "-" * 53)

    labels = [
        ("E1", "Short-term forecasting"),
        ("E2", "Long-term spectral"),
        ("E3", "Medium-noise reconstruction"),
        ("E4", "Medium-noise long-term forecast"),
        ("E5", "High-noise reconstruction"),
        ("E6", "High-noise long-term forecast"),
        ("E7", "Limited data, short-term"),
        ("E8", "Limited data, long-term"),
        ("E9", "Limited noisy data, short-term"),
        ("E10", "Limited noisy data, long-term"),
        ("E11", "Parametric interpolation"),
        ("E12", "Parametric extrapolation"),
    ]
    for key, desc in labels:
        print(f"  {key:<8} {desc:<35} {scores[key]:>8.2f}")
    print("  " + "-" * 53)
    print(f"  {'AVG':<8} {'Average (E1-E12)':<35} {scores['avg']:>8.2f}")

    print(f"\n  Total wall time: {wall_time:.0f}s")

    print("\n  CTF Published Leaderboard (avg):")
    for name, avg in CTF_LEADERBOARD.items():
        marker = " <--" if name in ("SINDy", "PyKoopman") else ""
        print(f"    {name:<15} {avg:>8.2f}{marker}")

    # --- Save outputs ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": "EML-Koopman",
        "config": {
            "n_observables": n_observables,
            "tree_depth": tree_depth,
            "n_epochs": n_epochs,
            "lr": lr,
        },
        "scores": scores,
        "wall_time_s": wall_time,
        "leaderboard": CTF_LEADERBOARD,
    }
    with open(out / "full_ctf_scores.json", "w") as f:
        json.dump(summary, f, indent=2)

    try:
        radar_plot(
            {"EML-Koopman": scores},
            title="EML-Koopman CTF E1-E12",
            save_path=str(out / "radar_e1_e12.png"),
        )
        print(f"\n  Radar plot saved to {out / 'radar_e1_e12.png'}")
    except Exception as exc:
        print(f"\n  (radar plot skipped: {exc})")

    print(f"  Full results saved to {out}")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Full CTF E1-E12 evaluation")
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="results/ctf_lorenz/full_ctf")
    args = parser.parse_args()

    run_full_ctf(
        n_observables=args.n_obs,
        tree_depth=args.depth,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
