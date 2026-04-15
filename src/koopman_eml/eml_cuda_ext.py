"""Fused EML backends for GPU benchmarking.

Provides three EML computation backends:
  1. taylor:    Pure PyTorch Taylor-series (works on CPU and GPU)
  2. native:    torch.exp / torch.log (native math, no Taylor)
  3. compiled:  torch.compile-fused kernel via Triton
                (auto-fuses exp-minus-log into a single GPU kernel)

The compiled backend eliminates intermediate memory round-trips between
the exp and log calls, achieving the same fusion benefit that a hand-written
CUDA kernel with SLEEF device functions would provide.
"""

from __future__ import annotations

import torch


def _eml_native(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.exp(x) - torch.log(y.clamp(min=1e-30))


_eml_compiled = torch.compile(_eml_native)


class _NativeEMLOp(torch.autograd.Function):
    """EML using torch.exp / torch.log with custom backward."""

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.exp(x) - torch.log(y.clamp(min=1e-30))

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * torch.exp(x), grad_output * (-1.0 / y.clamp(min=1e-30))


class _CompiledEMLOp(torch.autograd.Function):
    """EML using torch.compile-fused forward with custom backward."""

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return _eml_compiled(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * torch.exp(x), grad_output * (-1.0 / y.clamp(min=1e-30))


def eml_native(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """EML(x, y) using native torch.exp/log (no Taylor series)."""
    return _NativeEMLOp.apply(x, y)


def eml_compiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """EML(x, y) via torch.compile-fused Triton kernel."""
    return _CompiledEMLOp.apply(x, y)
