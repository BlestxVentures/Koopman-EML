"""
Adapter for the dynamicsai.org Common Task Framework (CTF).

Loads CTF datasets (ODE_Lorenz, PDE_KS) and evaluates predictions against
the 12-metric framework (E1-E12) from Wyder et al. (NeurIPS 2025).

Metrics:
    E1-E2:   Forecasting (short-term RMSE, long-term spectral)
    E3-E6:   Noisy data (medium/high noise reconstruction + forecasting)
    E7-E10:  Limited data (noise-free and noisy, short + long term)
    E11-E12: Parametric generalization (interpolation + extrapolation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import ctf4science
    _HAS_CTF = True
except ImportError:
    _HAS_CTF = False


# ---------------------------------------------------------------------------
# CTF metrics (self-contained implementation of the scoring formulas)
# ---------------------------------------------------------------------------


def short_term_score(prediction: np.ndarray, ground_truth: np.ndarray, k: Optional[int] = None) -> float:
    """E_ST = 100 * (1 - RMSE(pred[:k], truth[:k]) / ||truth[:k]||).

    Corresponds to the "weather forecast" metric in the CTF paper (Eq. 2).
    """
    if k is not None:
        prediction = prediction[:k]
        ground_truth = ground_truth[:k]
    residual = np.linalg.norm(prediction - ground_truth)
    baseline = np.linalg.norm(ground_truth)
    if baseline < 1e-30:
        return 0.0
    return float(100.0 * (1.0 - residual / baseline))


def long_term_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    k_max: int = 100,
) -> float:
    """E_LT = 100 * (1 - spectral_error).

    Corresponds to the "climate forecast" metric in the CTF paper (Eq. 3).
    Compares log power spectra over the first k_max wavenumbers.
    """
    def _log_power(x: np.ndarray, kmax: int) -> np.ndarray:
        spectra = []
        for col in range(x.shape[1] if x.ndim > 1 else 1):
            signal = x[:, col] if x.ndim > 1 else x
            fft_vals = np.fft.fftshift(np.fft.fft(signal))
            n = len(fft_vals)
            mid = n // 2
            lo, hi = max(mid - kmax, 0), min(mid + kmax + 1, n)
            spectra.append(np.log(np.abs(fft_vals[lo:hi]) ** 2 + 1e-30))
        return np.concatenate(spectra)

    p_pred = _log_power(prediction, k_max)
    p_truth = _log_power(ground_truth, k_max)
    residual = np.linalg.norm(p_pred - p_truth)
    baseline = np.linalg.norm(p_truth)
    if baseline < 1e-30:
        return 0.0
    return float(100.0 * (1.0 - residual / baseline))


def evaluate_ctf(
    results: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    k_max: int = 100,
) -> dict[str, float]:
    """Compute E1-E12 scores given prediction/truth arrays keyed by test ID.

    Expected keys follow CTF convention:
        X1pred (forecasting), X2pred/X3pred (noisy reconstruction/forecast), etc.
    Any missing keys are scored as 0.

    Returns a dict {"E1": ..., "E2": ..., ..., "E12": ..., "avg": ...}.
    """
    def _st(pred_key: str, truth_key: str) -> float:
        if pred_key in results and truth_key in truths:
            return short_term_score(results[pred_key], truths[truth_key])
        return 0.0

    def _lt(pred_key: str, truth_key: str) -> float:
        if pred_key in results and truth_key in truths:
            return long_term_score(results[pred_key], truths[truth_key], k_max=k_max)
        return 0.0

    scores = {
        "E1": _st("X1pred", "X1test"),
        "E2": _lt("X1pred", "X1test"),
        "E3": _st("X2pred", "X2test"),
        "E4": _lt("X3pred", "X3test"),
        "E5": _st("X4pred", "X4test"),
        "E6": _lt("X5pred", "X5test"),
        "E7": _st("X6pred", "X6test"),
        "E8": _lt("X6pred", "X6test"),
        "E9": _st("X7pred", "X7test"),
        "E10": _lt("X7pred", "X7test"),
        "E11": _st("X8pred", "X8test"),
        "E12": _st("X9pred", "X9test"),
    }
    scores["avg"] = float(np.mean([scores[f"E{i}"] for i in range(1, 13)]))
    return scores


# ---------------------------------------------------------------------------
# Radar plot
# ---------------------------------------------------------------------------


def radar_plot(
    scores_dict: dict[str, dict[str, float]],
    title: str = "CTF E1-E12 Comparison",
    save_path: Optional[str] = None,
):
    """Generate a 12-axis radar plot comparing multiple methods.

    Parameters
    ----------
    scores_dict : dict mapping method name -> {E1: ..., E2: ..., ...}
    title : str
    save_path : optional file path to save the figure
    """
    import matplotlib.pyplot as plt

    labels = [f"E{i}" for i in range(1, 13)]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_title(title, pad=20, fontsize=14)

    for method_name, scores in scores_dict.items():
        values = [scores.get(label, 0.0) for label in labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=method_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def load_ctf_lorenz(data_dir: Optional[str] = None) -> dict[str, np.ndarray]:
    """Load the CTF Lorenz dataset.

    Tries ctf4science package first; falls back to loading .npy files from data_dir.
    """
    if _HAS_CTF:
        try:
            from ctf4science.data import load_dataset
            return load_dataset("ODE_Lorenz")
        except Exception:
            pass

    if data_dir is None:
        data_dir = "data/ctf_lorenz"
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"CTF Lorenz data not found at {p}.  Install ctf4science or download "
            "from https://github.com/CTF-for-Science/ctf4science"
        )
    data = {}
    for f in p.glob("*.npy"):
        data[f.stem] = np.load(f)
    return data
