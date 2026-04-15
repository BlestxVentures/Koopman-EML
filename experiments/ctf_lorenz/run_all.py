"""Run all CTF Lorenz experiments sequentially and print comparison."""

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
    koopman_eigendecomposition,
    prediction_rollout,
)
from koopman_eml.ctf import long_term_score, short_term_score
from koopman_eml.training import train_koopman_eml
from baselines.edmd import EDMDModel
from baselines.deep_koopman import DeepKoopman


def _save(name: str, metrics: dict, pred: np.ndarray, base: str = "results/ctf_lorenz"):
    out = Path(base) / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "predictions.npy", pred)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def run_eml(data: dict, device: str) -> dict:
    print("\n" + "=" * 70)
    print("  EML-Koopman")
    print("=" * 70)
    X_k_np, X_k1_np = data["X_k_train"], data["X_k1_train"]
    rng = np.random.default_rng(0)
    if len(X_k_np) > 20000:
        idx = rng.choice(len(X_k_np), 20000, replace=False)
        X_k_np, X_k1_np = X_k_np[idx], X_k1_np[idx]

    X_k = torch.tensor(X_k_np, dtype=torch.float32)
    X_k1 = torch.tensor(X_k1_np, dtype=torch.float32)

    model = KoopmanEML(state_dim=3, n_observables=16, tree_depth=2, exp_order=10, ln_order=12)
    t0 = time.time()
    history = train_koopman_eml(
        model, X_k, X_k1,
        n_epochs=1200, lr=3e-3, device=device, verbose=True, batch_size=2048,
    )
    train_time = time.time() - t0

    formulas = extract_eml_formulas(model)
    print("\nSnapped formulas:")
    for i, f in enumerate(formulas[:8]):
        print(f"  g_{i} = {f}")

    x0 = torch.tensor(data["X1train"][-1], dtype=torch.float32)
    n_fc = len(data["X1test"])
    traj = prediction_rollout(model, x0, n_fc - 1, device=device)
    pred = traj[1:].numpy()
    truth = data["X1test"][:len(pred)]

    m = compute_metrics(pred, truth)
    m["E1"] = short_term_score(pred, truth)
    m["E2"] = long_term_score(pred, truth)
    m["train_time"] = train_time
    m["n_params"] = sum(p.numel() for p in model.parameters())
    _save("eml", m, pred)

    # Also save model and formulas
    out = Path("results/ctf_lorenz/eml")
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "formulas.json", "w") as f:
        json.dump(formulas, f, indent=2)

    return m


def run_edmd(data: dict, dictionary: str) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  EDMD-{dictionary.upper()}")
    print("=" * 70)
    X_k, X_k1 = data["X_k_train"], data["X_k1_train"]
    if len(X_k) > 20000:
        idx = np.random.default_rng(0).choice(len(X_k), 20000, replace=False)
        X_k, X_k1 = X_k[idx], X_k1[idx]

    kwargs = {"poly_degree": 4} if dictionary == "poly" else {"n_rbf": 200, "rbf_sigma": 1.0}
    model = EDMDModel(dictionary=dictionary, **kwargs)
    t0 = time.time()
    model.fit(X_k, X_k1)
    train_time = time.time() - t0

    x0 = data["X1train"][-1]
    n_fc = len(data["X1test"])
    pred = model.predict(x0, n_fc - 1)[1:]
    truth = data["X1test"][:len(pred)]

    # Clip divergent predictions
    pred = np.clip(pred, -100, 100)

    m = compute_metrics(pred, truth)
    m["E1"] = short_term_score(pred, truth)
    m["E2"] = long_term_score(pred, truth)
    m["train_time"] = train_time
    _save(f"edmd_{dictionary}", m, pred)
    return m


def run_deep_koopman(data: dict, device: str) -> dict:
    print("\n" + "=" * 70)
    print("  Deep Koopman (autoencoder)")
    print("=" * 70)
    X_k_np, X_k1_np = data["X_k_train"], data["X_k1_train"]
    if len(X_k_np) > 20000:
        idx = np.random.default_rng(0).choice(len(X_k_np), 20000, replace=False)
        X_k_np, X_k1_np = X_k_np[idx], X_k1_np[idx]

    X_k = torch.tensor(X_k_np, dtype=torch.float32).to(device)
    X_k1 = torch.tensor(X_k1_np, dtype=torch.float32).to(device)

    model = DeepKoopman(state_dim=3, latent_dim=32, hidden_dim=128, n_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1200)

    t0 = time.time()
    for epoch in range(1200):
        optimizer.zero_grad()
        out = model(X_k, X_k1)
        loss = out["pred_loss"] + out["recon_loss"] + 1e-4 * out["reg_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if epoch % 300 == 0 or epoch == 1199:
            print(f"  [{epoch:5d}/1200] pred={out['pred_loss']:.6f} recon={out['recon_loss']:.6f}")
    train_time = time.time() - t0

    model.eval()
    x0 = torch.tensor(data["X1train"][-1], dtype=torch.float32).to(device)
    n_fc = len(data["X1test"])
    trajectory = [x0.unsqueeze(0)]
    with torch.no_grad():
        g = model.encode(x0.unsqueeze(0))
        for _ in range(n_fc - 1):
            g = model.K(g)
            trajectory.append(model.decode(g))
    pred = torch.cat(trajectory[1:], dim=0).cpu().numpy()
    truth = data["X1test"][:len(pred)]

    pred = np.clip(pred, -100, 100)

    m = compute_metrics(pred, truth)
    m["E1"] = short_term_score(pred, truth)
    m["E2"] = long_term_score(pred, truth)
    m["train_time"] = train_time
    m["n_params"] = sum(p.numel() for p in model.parameters())
    _save("deep_koopman", m, pred)
    return m


CTF_LEADERBOARD = {
    "LSTM (pub.)": {"E1": 64.54, "note": "CTF avg"},
    "DeepONet (pub.)": {"E1": 57.80, "note": "CTF avg"},
    "Reservoir (pub.)": {"E1": 54.87, "note": "CTF avg"},
    "KAN (pub.)": {"E1": 47.28, "note": "CTF avg"},
    "SINDy (pub.)": {"E1": -3.00, "note": "CTF avg"},
    "PyKoopman (pub.)": {"E1": -20.11, "note": "CTF avg"},
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = generate_lorenz_trajectories()
    print(f"Train pairs: {data['X_k_train'].shape}, Test forecast: {data['X1test'].shape}")

    results = {}
    results["EML-Koopman"] = run_eml(data, device)
    results["EDMD-Poly"] = run_edmd(data, "poly")
    results["EDMD-RBF"] = run_edmd(data, "rbf")
    results["Deep Koopman"] = run_deep_koopman(data, device)

    # Final summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY: CTF Lorenz")
    print("=" * 70)
    print(f"{'Method':<22} {'E1 (short)':>10} {'E2 (long)':>10} {'RMSE':>10} {'Valid Steps':>12} {'Time (s)':>9}")
    print("-" * 75)

    for name, m in results.items():
        print(f"{name:<22} {m.get('E1', 0):>10.2f} {m.get('E2', 0):>10.2f} "
              f"{m.get('rmse', 0):>10.4f} {m.get('valid_prediction_steps', 0):>12} "
              f"{m.get('train_time', 0):>9.1f}")

    print("-" * 75)
    print("CTF Published Leaderboard (avg across E1-E12):")
    for name, info in CTF_LEADERBOARD.items():
        print(f"  {name:<22} avg={info['E1']:.2f}")

    # Save combined summary
    out = Path("results/ctf_lorenz")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
