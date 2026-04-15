"""Run Deep Koopman baseline on CTF Lorenz."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines.deep_koopman import DeepKoopman
from koopman_eml.ctf import short_term_score
from koopman_eml.analysis import compute_metrics
from experiments.ctf_lorenz.generate_data import generate_lorenz_trajectories


def run(
    latent_dim: int = 32,
    n_epochs: int = 2000,
    lr: float = 1e-3,
    device: str = "auto",
    output_dir: str = "results/ctf_lorenz/deep_koopman",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Deep Koopman on CTF Lorenz")
    print("=" * 70)

    data = generate_lorenz_trajectories()
    X_k = torch.tensor(data["X_k_train"], dtype=torch.float32).to(device)
    X_k1 = torch.tensor(data["X_k1_train"], dtype=torch.float32).to(device)

    model = DeepKoopman(state_dim=3, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(X_k, X_k1)
        loss = out["pred_loss"] + out["recon_loss"] + 1e-4 * out["reg_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if epoch % 400 == 0 or epoch == n_epochs - 1:
            print(f"[{epoch:5d}/{n_epochs}] pred={out['pred_loss']:.6f} recon={out['recon_loss']:.6f}")

    # Forecast
    model.eval()
    x0 = torch.tensor(data["X1train"][-1], dtype=torch.float32).to(device)
    n_forecast = len(data["X1test"])
    trajectory = [x0.unsqueeze(0)]
    with torch.no_grad():
        g = model.encode(x0.unsqueeze(0))
        for _ in range(n_forecast - 1):
            g = model.K(g)
            trajectory.append(model.decode(g))
    pred = torch.cat(trajectory[1:], dim=0).cpu().numpy()
    truth = data["X1test"]

    n_eval = min(len(pred), len(truth))
    pred, truth = pred[:n_eval], truth[:n_eval]

    metrics = compute_metrics(pred, truth)
    e1 = short_term_score(pred, truth)
    print(f"\nRMSE={metrics['rmse']:.4f}, valid_steps={metrics['valid_prediction_steps']}, E1={e1:.2f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path / "model.pt")
    np.save(out_path / "predictions.npy", pred)
    with open(out_path / "metrics.json", "w") as f:
        json.dump({**metrics, "E1": e1}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run()
