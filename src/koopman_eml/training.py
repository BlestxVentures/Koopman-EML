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
import torch.nn as nn
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
    phase1_frac: float = 0.6,
    phase2_frac: float = 0.3,
    batch_size: Optional[int] = None,
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Train a KoopmanEML model on consecutive state-pair data.

    The three training phases are controlled by ``phase1_frac`` (soft
    exploration) and ``phase2_frac`` (hardening).  The remainder
    ``1 - phase1_frac - phase2_frac`` is the snap + fine-tune phase.
    Increasing ``phase1_frac`` and/or ``tau_start`` gives the model
    more exploration time, which helps when the candidate set is large.
    """
    model = model.to(device)
    X_k = X_k.to(device)
    X_k1 = X_k1.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history: dict[str, list[float]] = {"total": [], "pred": [], "recon": [], "tau": []}

    phase1_end = int(phase1_frac * n_epochs)
    phase2_end = int((phase1_frac + phase2_frac) * n_epochs)
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


def _expand_logits_for_complex(model: KoopmanEML) -> None:
    """Expand real-trained logits to include complex candidate slots.

    Inserts zero-initialized logit columns for the new `i` / `ix_j`
    candidates so the routing structure learned during real pre-training
    is preserved.  Only affects non-leaf levels' child-slot position
    (it shifts to the end of the expanded vector).
    """
    dict_mod = model.dictionary
    n_vars = dict_mod.n_vars
    n_complex_new = (1 if dict_mod.use_complex else 0) + (
        n_vars if dict_mod.allow_imaginary_vars else 0
    )
    if n_complex_new == 0:
        return

    with torch.no_grad():
        for level in range(dict_mod.depth):
            old = dict_mod.level_logits[level]
            is_leaf = level == 0
            n_trees, n_nodes, n_old_choices, two = old.shape
            n_base_old = n_old_choices - (0 if is_leaf else 1)

            new_cols = torch.zeros(
                n_trees, n_nodes, n_complex_new, two,
                device=old.device, dtype=old.dtype,
            )

            if is_leaf:
                expanded = torch.cat([
                    old[:, :, :1, :],
                    new_cols,
                    old[:, :, 1:, :],
                ], dim=2)
            else:
                expanded = torch.cat([
                    old[:, :, :1, :],
                    new_cols,
                    old[:, :, 1:n_base_old, :],
                    old[:, :, n_base_old:, :],
                ], dim=2)

            dict_mod.level_logits[level] = nn.Parameter(expanded)


def train_warmstart_complex(
    model: KoopmanEML,
    X_k: torch.Tensor,
    X_k1: torch.Tensor,
    *,
    pretrain_epochs: int = 800,
    complex_epochs: int = 400,
    lr: float = 3e-3,
    batch_size: int | None = 2048,
    device: str = "cpu",
    verbose: bool = True,
    **train_kwargs,
) -> dict:
    """Two-stage warm-start: train a real proxy, then expand to complex.

    Stage 1: Build a temporary real-only ``KoopmanEML`` with matching
    architecture, train for ``pretrain_epochs``.
    Stage 2: Copy trained real logits into the complex model (expanding
    with zero-initialized complex slots), then train the full complex
    model for ``complex_epochs``.
    """
    from koopman_eml.eml_tree import EMLTreeVectorized  # avoid circular at top

    cplx_dict = model.dictionary
    depth = cplx_dict.depth
    n_vars = cplx_dict.n_vars
    n_obs = model.n_obs

    if verbose:
        print("[Warm-start] Stage 1: real pre-training")

    real_model = KoopmanEML(
        state_dim=model.state_dim, n_observables=n_obs,
        tree_depth=depth, exp_order=model.exp_order, ln_order=model.ln_order,
        use_complex=False, clamp_input=model.clamp_input,
    )

    h1 = train_koopman_eml(
        real_model, X_k, X_k1,
        n_epochs=pretrain_epochs, lr=lr, batch_size=batch_size,
        device=device, verbose=verbose, **train_kwargs,
    )

    if verbose:
        print("\n[Warm-start] Stage 2: expanding grammar to complex")

    dev = next(real_model.parameters()).device
    with torch.no_grad():
        for level in range(depth):
            real_logits = real_model.dictionary.level_logits[level].data
            cplx_target = cplx_dict.level_logits[level]
            n_trees, n_nodes, _, two = real_logits.shape
            is_leaf = level == 0
            n_real_base = real_logits.shape[2] - (0 if is_leaf else 1)
            n_cplx_base = cplx_target.shape[2] - (0 if is_leaf else 1)

            new_logits = torch.zeros_like(cplx_target.data)
            new_logits[:, :, :1, :] = real_logits[:, :, :1, :]
            new_logits[:, :, (n_cplx_base - n_vars):(n_cplx_base - n_vars + n_vars), :] = \
                real_logits[:, :, 1:1 + n_vars, :]
            if not is_leaf:
                new_logits[:, :, -1, :] = real_logits[:, :, -1, :]
            cplx_target.data.copy_(new_logits)

        if hasattr(model, "C_re"):
            model.C_re.weight.data.copy_(real_model.C.weight.data.to(dev))
            nn.init.zeros_(model.C_im.weight)

        K_real = real_model.K.data.to(dev)
        model.K.data.copy_(
            K_real.to(torch.cfloat)
            + 1j * torch.randn_like(K_real) * 0.01
        )

    h2 = train_koopman_eml(
        model, X_k, X_k1,
        n_epochs=complex_epochs, lr=lr * 0.5, batch_size=batch_size,
        device=device, verbose=verbose, **train_kwargs,
    )

    combined = {k: h1[k] + h2[k] for k in h1}
    return combined


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
