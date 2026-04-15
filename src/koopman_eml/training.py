"""
Three-phase training loop with Gumbel-softmax temperature annealing.

Phase 1 (60%): Soft exploration -- tau anneals from tau_start to 1.0
Phase 2 (30%): Hardening -- tau anneals from 1.0 to tau_end
Phase 3 (10%): Snap weights to one-hot, freeze trees, fine-tune K and C
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.utils

from koopman_eml.koopman_model import KoopmanEML


def train_koopman_eml(
    model: KoopmanEML,
    X_k: torch.Tensor,
    X_k1: torch.Tensor,
    *,
    n_epochs: int = 2000,
    lr: float = 1e-3,
    lambda_recon: float = 1.0,
    mu_reg: float = 1e-4,
    tau_start: float = 2.0,
    tau_end: float = 0.1,
    batch_size: Optional[int] = None,
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Train a KoopmanEML model on consecutive state-pair data."""
    model = model.to(device)
    X_k = X_k.to(device)
    X_k1 = X_k1.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history: dict[str, list[float]] = {"total": [], "pred": [], "recon": [], "tau": []}

    phase1_end = int(0.6 * n_epochs)
    phase2_end = int(0.9 * n_epochs)
    n_samples = X_k.shape[0]

    for epoch in range(n_epochs):
        # --- Temperature schedule ---
        if epoch < phase1_end:
            progress = epoch / phase1_end
            tau = tau_start * (1.0 / tau_start) ** progress
        elif epoch < phase2_end:
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            tau = 1.0 * (tau_end / 1.0) ** progress
        else:
            tau = 0.01
            if epoch == phase2_end:
                if verbose:
                    print(f"\n[Epoch {epoch}] Snapping EML tree weights...")
                _snap_dictionary(model)
                finetune_params: list[torch.nn.Parameter] = [model.K]
                if hasattr(model, "C"):
                    finetune_params.extend(model.C.parameters())
                if hasattr(model, "C_re"):
                    finetune_params.extend(model.C_re.parameters())
                if hasattr(model, "C_im"):
                    finetune_params.extend(model.C_im.parameters())
                optimizer = torch.optim.Adam(finetune_params, lr=lr * 0.1)

        # --- Mini-batch or full-batch ---
        if batch_size is not None and batch_size < n_samples:
            idx = torch.randperm(n_samples, device=device)[:batch_size]
            xk_batch, xk1_batch = X_k[idx], X_k1[idx]
        else:
            xk_batch, xk1_batch = X_k, X_k1

        optimizer.zero_grad()
        out = model(xk_batch, xk1_batch, tau=tau)
        loss = out["pred_loss"] + lambda_recon * out["recon_loss"] + mu_reg * out["reg_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        history["total"].append(loss.item())
        history["pred"].append(out["pred_loss"].item())
        history["recon"].append(out["recon_loss"].item())
        history["tau"].append(tau)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            print(
                f"[{epoch:5d}/{n_epochs}]  tau={tau:.4f}  "
                f"pred={out['pred_loss']:.6f}  recon={out['recon_loss']:.6f}  "
                f"total={loss:.6f}"
            )

        if checkpoint_dir and (epoch + 1) % 500 == 0:
            ckpt_path = Path(checkpoint_dir) / f"koopman_eml_epoch{epoch + 1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    return history


def _snap_dictionary(model: KoopmanEML) -> None:
    """Hard-snap all tree routing logits to argmax and freeze them."""
    for level_logits in model.dictionary.level_logits:
        level_logits.requires_grad_(False)
        with torch.no_grad():
            for t in range(level_logits.shape[0]):
                for n in range(level_logits.shape[1]):
                    for side in range(2):
                        idx = level_logits[t, n, :, side].argmax()
                        level_logits[t, n, :, side] = -10.0
                        level_logits[t, n, idx, side] = 10.0
