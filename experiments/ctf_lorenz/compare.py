"""Load all experiment results and produce unified comparison tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from koopman_eml.ctf import radar_plot


CTF_LEADERBOARD_LORENZ = {
    "LSTM": 64.54,
    "DeepONet": 57.80,
    "Reservoir": 54.87,
    "KAN": 47.28,
    "ODE-LSTM": 41.67,
    "SINDy": -3.00,
    "PyKoopman": -20.11,
}


def load_metrics(results_dir: str) -> dict[str, dict]:
    """Load metrics.json from each method's results directory."""
    base = Path(results_dir)
    methods = {}
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir():
            metrics_file = subdir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    methods[subdir.name] = json.load(f)
    return methods


def comparison_table(methods: dict[str, dict]) -> str:
    """Generate a markdown comparison table."""
    lines = ["| Method | RMSE | Valid Steps | E1 (short-term) |", "|--------|------|-------------|-----------------|"]
    for name, m in sorted(methods.items(), key=lambda x: -x[1].get("E1", 0)):
        lines.append(
            f"| {name} | {m.get('rmse', '-'):.4f} | "
            f"{m.get('valid_prediction_steps', '-')} | "
            f"{m.get('E1', '-'):.2f} |"
        )

    lines.append("")
    lines.append("### CTF Leaderboard (published, Lorenz avg)")
    lines.append("| Method | Avg Score |")
    lines.append("|--------|-----------|")
    for name, score in sorted(CTF_LEADERBOARD_LORENZ.items(), key=lambda x: -x[1]):
        lines.append(f"| {name} | {score:.2f} |")

    return "\n".join(lines)


def bar_chart(methods: dict[str, dict], save_path: str = "paper/figures/e1_comparison.png"):
    """Bar chart comparing E1 scores."""
    names = list(methods.keys())
    e1_scores = [m.get("E1", 0) for m in methods.values()]

    for lb_name, lb_score in CTF_LEADERBOARD_LORENZ.items():
        names.append(f"{lb_name} (CTF)")
        e1_scores.append(lb_score)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2d5a27"] * len(methods) + ["#8b3a3a"] * len(CTF_LEADERBOARD_LORENZ)
    bars = ax.barh(names, e1_scores, color=colors)
    ax.set_xlabel("E1 Score (short-term forecasting)")
    ax.set_title("CTF Lorenz: EML-Koopman vs Baselines vs Published")
    ax.axvline(x=0, color="black", linewidth=0.5)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    return fig


def main(results_dir: str = "results/ctf_lorenz"):
    methods = load_metrics(results_dir)
    if not methods:
        print(f"No results found in {results_dir}. Run experiments first.")
        return

    table = comparison_table(methods)
    print(table)

    out = Path("paper/tables")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.md", "w") as f:
        f.write(table)

    bar_chart(methods)
    print("\nComparison complete.")


if __name__ == "__main__":
    main()
