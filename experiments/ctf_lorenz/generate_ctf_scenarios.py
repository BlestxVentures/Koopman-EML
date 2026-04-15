"""Generate all data scenarios required for CTF E1-E12 evaluation.

Produces a single dict with keys following CTF conventions:
    X1train/X1test  -- clean forecasting           (E1, E2)
    X2train/X2test  -- medium-noise reconstruction  (E3)
    X3train/X3test  -- medium-noise forecasting     (E4)
    X4train/X4test  -- high-noise reconstruction    (E5)
    X5train/X5test  -- high-noise forecasting       (E6)
    X6train/X6test  -- limited clean data           (E7, E8)
    X7train/X7test  -- limited noisy data           (E9, E10)
    X8train/X8test  -- parametric interpolation     (E11)
    X9train/X9test  -- parametric extrapolation     (E12)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def _integrate_lorenz(
    n_trajectories: int,
    n_steps: int,
    dt: float,
    sigma: float,
    rho: float,
    beta: float,
    rng: np.random.Generator,
    ic_scale: float = 1.0,
) -> list[np.ndarray]:
    t_span = (0, n_steps * dt)
    t_eval = np.linspace(0, n_steps * dt, n_steps + 1)
    trajectories = []
    for _ in range(n_trajectories):
        ic = rng.standard_normal(3) * ic_scale + np.array([1.0, 1.0, 25.0])
        sol = solve_ivp(
            lorenz, t_span, ic, t_eval=t_eval,
            args=(sigma, rho, beta), rtol=1e-10, atol=1e-12,
        )
        trajectories.append(sol.y.T)
    return trajectories


def _normalize(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std[std < 1e-8] = 1.0
    return (states - mean) / std, mean, std


def _add_noise(data: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise at a specified SNR (in dB) to each column independently."""
    noisy = data.copy()
    for col in range(data.shape[1]):
        signal_power = np.mean(data[:, col] ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noisy[:, col] += rng.normal(0, np.sqrt(noise_power), size=data.shape[0])
    return noisy


def generate_all_ctf_scenarios(
    n_trajectories: int = 20,
    n_steps: int = 5000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    medium_noise_snr: float = 20.0,
    high_noise_snr: float = 10.0,
    limited_data_fraction: float = 0.1,
    param_interp_rhos: tuple[float, ...] = (26.0, 30.0),
    param_interp_test_rho: float = 28.0,
    param_extrap_train_rhos: tuple[float, ...] = (26.0, 28.0),
    param_extrap_test_rho: float = 32.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # --- Base trajectories (E1-E2) ---
    trajectories = _integrate_lorenz(n_trajectories, n_steps, dt, sigma, rho, beta, rng)
    all_states = np.concatenate(trajectories, axis=0)
    states_norm, mean, std = _normalize(all_states)

    n_total = len(states_norm)
    split = int(0.8 * n_total)
    X_train_full = states_norm[:split]
    X_test_full = states_norm[split:]

    X_k_train = X_train_full[:-1]
    X_k1_train = X_train_full[1:]
    X_k_test = X_test_full[:-1]
    X_k1_test = X_test_full[1:]

    train_len = int(0.9 * len(trajectories[0]))
    forecast_norm = (trajectories[0] - mean) / std
    X1train = forecast_norm[:train_len]
    X1test = forecast_norm[train_len:]

    data: dict[str, np.ndarray] = {
        "X_k_train": X_k_train,
        "X_k1_train": X_k1_train,
        "X_k_test": X_k_test,
        "X_k1_test": X_k1_test,
        "X1train": X1train,
        "X1test": X1test,
        "mean": mean,
        "std": std,
        "dt": np.array(dt),
    }

    # --- Medium noise (E3, E4) ---
    X2_noisy = _add_noise(X1train, medium_noise_snr, rng)
    data["X2train"] = X2_noisy
    data["X2test"] = X1train.copy()
    X3_noisy_forecast = _add_noise(forecast_norm, medium_noise_snr, rng)
    data["X3train"] = X3_noisy_forecast[:train_len]
    data["X3test"] = X1test.copy()

    # --- High noise (E5, E6) ---
    X4_noisy = _add_noise(X1train, high_noise_snr, rng)
    data["X4train"] = X4_noisy
    data["X4test"] = X1train.copy()
    X5_noisy_forecast = _add_noise(forecast_norm, high_noise_snr, rng)
    data["X5train"] = X5_noisy_forecast[:train_len]
    data["X5test"] = X1test.copy()

    # --- Limited data, clean (E7, E8) ---
    n_limited = max(int(len(X_k_train) * limited_data_fraction), 100)
    idx_limited = rng.choice(len(X_k_train), n_limited, replace=False)
    data["X6_k_train"] = X_k_train[idx_limited]
    data["X6_k1_train"] = X_k1_train[idx_limited]
    data["X6train"] = X1train.copy()
    data["X6test"] = X1test.copy()

    # --- Limited data, noisy (E9, E10) ---
    X7_noisy_pairs = _add_noise(X_k_train[idx_limited], medium_noise_snr, rng)
    X7_noisy_pairs1 = _add_noise(X_k1_train[idx_limited], medium_noise_snr, rng)
    data["X7_k_train"] = X7_noisy_pairs
    data["X7_k1_train"] = X7_noisy_pairs1
    X7_noisy_train = _add_noise(X1train, medium_noise_snr, rng)
    data["X7train"] = X7_noisy_train
    data["X7test"] = X1test.copy()

    # --- Parametric interpolation (E11) ---
    interp_trajs = []
    for r in param_interp_rhos:
        interp_trajs.extend(
            _integrate_lorenz(n_trajectories // 2, n_steps, dt, sigma, r, beta, rng)
        )
    interp_all = np.concatenate(interp_trajs, axis=0)
    interp_norm = (interp_all - mean) / std
    data["X8_k_train"] = interp_norm[:-1]
    data["X8_k1_train"] = interp_norm[1:]

    interp_test_trajs = _integrate_lorenz(2, n_steps, dt, sigma, param_interp_test_rho, beta, rng)
    interp_test = np.concatenate(interp_test_trajs, axis=0)
    interp_test_norm = (interp_test - mean) / std
    test_len = int(0.9 * len(interp_test_trajs[0]))
    data["X8train"] = interp_test_norm[:test_len]
    data["X8test"] = interp_test_norm[test_len:]

    # --- Parametric extrapolation (E12) ---
    extrap_trajs = []
    for r in param_extrap_train_rhos:
        extrap_trajs.extend(
            _integrate_lorenz(n_trajectories // 2, n_steps, dt, sigma, r, beta, rng)
        )
    extrap_all = np.concatenate(extrap_trajs, axis=0)
    extrap_norm = (extrap_all - mean) / std
    data["X9_k_train"] = extrap_norm[:-1]
    data["X9_k1_train"] = extrap_norm[1:]

    extrap_test_trajs = _integrate_lorenz(2, n_steps, dt, sigma, param_extrap_test_rho, beta, rng)
    extrap_test = np.concatenate(extrap_test_trajs, axis=0)
    extrap_test_norm = (extrap_test - mean) / std
    test_len = int(0.9 * len(extrap_test_trajs[0]))
    data["X9train"] = extrap_test_norm[:test_len]
    data["X9test"] = extrap_test_norm[test_len:]

    return data


def save_ctf_data(data: dict[str, np.ndarray], output_dir: str = "data/ctf_lorenz_full") -> None:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    for key, arr in data.items():
        if isinstance(arr, np.ndarray):
            np.save(p / f"{key}.npy", arr)
    print(f"Saved {sum(1 for v in data.values() if isinstance(v, np.ndarray))} arrays to {p}")


if __name__ == "__main__":
    data = generate_all_ctf_scenarios()
    save_ctf_data(data)
    print("\nScenario shapes:")
    for key in sorted(data.keys()):
        if isinstance(data[key], np.ndarray):
            print(f"  {key:20s}  {str(data[key].shape):>20s}")
