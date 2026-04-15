"""
Taylor-series primitives and the vectorized EML operator.

EML(x, y) = exp(x) - ln(y)

Both exp and ln are evaluated via Horner-form polynomial approximations with
range reduction, so the entire EML call reduces to fused multiply-adds on GPU.
A NumPy-only variant (eml_numpy) is provided for PySINDy compatibility.

References:
    Odrzywołek (2026), "All elementary functions from a single operator"
    arXiv:2603.21852v2
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Real-valued Taylor primitives (PyTorch)
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)


def taylor_exp(x: torch.Tensor, order: int = 12) -> torch.Tensor:
    """exp(x) via Horner-form Taylor series with range reduction.

    Decompose x = k*ln2 + r (|r| <= ln2/2), evaluate exp(r) as a polynomial,
    then scale by 2^k.  Each Horner step is one FMA -- no transcendental calls.
    """
    k = torch.round(x / _LN2)
    r = x - k * _LN2

    result = torch.ones_like(r)
    for n in range(order, 0, -1):
        result = 1.0 + (r / n) * result

    return result * torch.pow(2.0, k)


def taylor_ln(y: torch.Tensor, order: int = 16) -> torch.Tensor:
    """ln(y) via range reduction + arctanh Horner series.

    Decompose y = m * 2^e with m in [0.5, 1).  Then
        ln(y) = e*ln2 + 2*arctanh((m-1)/(m+1))
    and the arctanh series converges rapidly because |u| <= 1/3.
    Requires y > 0.
    """
    e = torch.floor(torch.log2(y.abs().clamp(min=1e-45)))
    m = y * torch.pow(2.0, -e)

    mask = m < 0.5
    m = torch.where(mask, m * 2.0, m)
    e = torch.where(mask, e - 1.0, e)

    u = (m - 1.0) / (m + 1.0)
    u2 = u * u

    terms = order // 2
    s = torch.zeros_like(u)
    for k in range(terms, -1, -1):
        s = 1.0 / (2 * k + 1) + u2 * s
    s = u * s

    return e * _LN2 + 2.0 * s


# ---------------------------------------------------------------------------
# Complex Taylor primitives (PyTorch)
# ---------------------------------------------------------------------------


def taylor_exp_complex(z: torch.Tensor, order: int = 12) -> torch.Tensor:
    """Complex exp(z) via Horner form.  Range reduction on real part only."""
    k = torch.round(z.real / _LN2)
    r = torch.complex(z.real - k * _LN2, z.imag)

    result = torch.ones_like(r)
    for n in range(order, 0, -1):
        result = 1.0 + (r / n) * result

    scale = torch.pow(2.0, k)
    return result * scale.to(z.dtype)


def taylor_ln_complex(w: torch.Tensor, order: int = 16) -> torch.Tensor:
    """Complex ln(w) = ln|w| + i*arg(w).  Principal branch."""
    modulus = w.abs().clamp(min=1e-45)
    ln_mod = taylor_ln(modulus, order=order)
    angle = torch.angle(w)
    return torch.complex(ln_mod, angle)


# ---------------------------------------------------------------------------
# Custom autograd EML operator
# ---------------------------------------------------------------------------


class _EMLOp(torch.autograd.Function):
    """EML(x, y) = taylor_exp(x) - taylor_ln(y) with custom backward."""

    @staticmethod
    def forward(ctx, x, y, exp_order, ln_order, use_complex):
        ctx.save_for_backward(x, y)
        ctx.exp_order = exp_order
        ctx.use_complex = use_complex

        if use_complex:
            exp_x = taylor_exp_complex(x, order=exp_order)
            ln_y = taylor_ln_complex(y, order=ln_order)
        else:
            exp_x = taylor_exp(x, order=exp_order)
            ln_y = taylor_ln(y.clamp(min=1e-30), order=ln_order)

        return exp_x - ln_y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors

        if ctx.use_complex:
            dfdx = taylor_exp_complex(x, order=ctx.exp_order)
        else:
            dfdx = taylor_exp(x, order=ctx.exp_order)

        dfdy = -1.0 / y.clamp(min=1e-30) if not ctx.use_complex else -1.0 / y

        return grad_output * dfdx, grad_output * dfdy, None, None, None


def eml(
    x: torch.Tensor,
    y: torch.Tensor,
    exp_order: int = 12,
    ln_order: int = 16,
    use_complex: bool = False,
    backend: str = "taylor",
) -> torch.Tensor:
    """Vectorized EML(x, y) = exp(x) - ln(y).

    Parameters
    ----------
    backend : {"taylor", "native", "compiled"}
        "taylor"   -- Horner-form Taylor series (default, works on CPU+GPU)
        "native"   -- torch.exp / torch.log (no Taylor, GPU benefits)
        "compiled" -- torch.compile-fused Triton kernel (best GPU perf)
    """
    if backend == "taylor":
        return _EMLOp.apply(x, y, exp_order, ln_order, use_complex)
    elif backend == "native":
        from koopman_eml.eml_cuda_ext import eml_native
        return eml_native(x, y)
    elif backend == "compiled":
        from koopman_eml.eml_cuda_ext import eml_compiled
        return eml_compiled(x, y)
    else:
        raise ValueError(f"Unknown EML backend: {backend!r}")


# ---------------------------------------------------------------------------
# NumPy-only EML (for PySINDy / non-autograd contexts)
# ---------------------------------------------------------------------------


def eml_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """EML(x, y) = exp(x) - ln(y) using NumPy.  No autograd overhead."""
    return np.exp(x) - np.log(np.clip(y, 1e-30, None))
