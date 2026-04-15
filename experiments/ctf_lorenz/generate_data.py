"""Generate Lorenz trajectory data for CTF-style experiments.

If the ctf4science package is available, loads the official CTF Lorenz dataset.
Otherwise, generates Lorenz trajectories from scratch using scipy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def generate_lorenz_trajectories(
    n_trajectories: int = 20,
    n_steps: int = 5000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    ic_scale: float = 1.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate multiple Lorenz trajectories from random initial conditions."""
    rng = np.random.default_rng(seed)
    t_span = (0, n_steps * dt)
    t_eval = np.linspace(0, n_steps * dt, n_steps + 1)

    all_states = []
    for _ in range(n_trajectories):
        ic = rng.standard_normal(3) * ic_scale + np.array([1.0, 1.0, 25.0])
        sol = solve_ivp(lorenz, t_span, ic, t_eval=t_eval, args=(sigma, rho, beta), rtol=1e-10, atol=1e-12)
        all_states.append(sol.y.T)

    states = np.concatenate(all_states, axis=0)
    n_total = len(states)

    # Train: first 80%, test: last 20%
    split = int(0.8 * n_total)
    X_train = states[:split]
    X_test = states[split:]

    # Consecutive pairs
    X_k_train = X_train[:-1]
    X_k1_train = X_train[1:]
    X_k_test = X_test[:-1]
    X_k1_test = X_test[1:]

    # CTF-style forecasting: train on long window, predict short window
    train_len = int(0.9 * len(all_states[0]))
    forecast_traj = all_states[0]
    X1train = forecast_traj[:train_len]
    X1test = forecast_traj[train_len:]

    return {
        "X_k_train": X_k_train,
        "X_k1_train": X_k1_train,
        "X_k_test": X_k_test,
        "X_k1_test": X_k1_test,
        "X1train": X1train,
        "X1test": X1test,
        "t_eval": t_eval,
        "dt": dt,
    }


def save_data(data: dict[str, np.ndarray], output_dir: str = "data/ctf_lorenz") -> None:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    for key, arr in data.items():
        if isinstance(arr, np.ndarray):
            np.save(p / f"{key}.npy", arr)
    print(f"Saved {len(data)} arrays to {p}")


if __name__ == "__main__":
    data = generate_lorenz_trajectories()
    save_data(data)
    print(f"Train pairs: {data['X_k_train'].shape}")
    print(f"Test pairs:  {data['X_k_test'].shape}")
    print(f"Forecast train: {data['X1train'].shape}")
    print(f"Forecast test:  {data['X1test'].shape}")
