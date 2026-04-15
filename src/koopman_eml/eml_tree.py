"""
Parameterized EML trees (master formula) for Koopman observable dictionaries.

Every elementary function can be expressed as a binary tree of identical
eml(x, y) = exp(x) - ln(y) nodes with grammar  S -> 1 | eml(S, S).
This module provides:
    - EMLNode: a single node with Gumbel-softmax routing
    - EMLTree: a recursive tree (debugging / single-tree use)
    - EMLTreeVectorized: batched multi-tree evaluation (production)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_eml.eml_ops import eml


class EMLNode(nn.Module):
    """Single parameterized EML node with Gumbel-softmax input routing.

    Each input is soft-selected from {1, x_0, ..., x_{n-1}} (leaves)
    or {1, x_0, ..., x_{n-1}, f_child} (interior nodes).
    """

    def __init__(self, is_leaf: bool = False, n_vars: int = 1):
        super().__init__()
        self.is_leaf = is_leaf
        self.n_vars = n_vars

        n_choices = 1 + n_vars + (0 if is_leaf else 1)
        self.left_logits = nn.Parameter(torch.randn(n_choices) * 0.1)
        self.right_logits = nn.Parameter(torch.randn(n_choices) * 0.1)

    def route(
        self,
        logits: torch.Tensor,
        const_one: torch.Tensor,
        x_vars: torch.Tensor,
        f_child: Optional[torch.Tensor],
        tau: float,
    ) -> torch.Tensor:
        weights = F.gumbel_softmax(logits, tau=tau, hard=False)
        candidates = [const_one]
        for i in range(self.n_vars):
            candidates.append(x_vars if x_vars.dim() == 1 else x_vars[:, i])
        if f_child is not None:
            candidates.append(f_child)
        stack = torch.stack(candidates, dim=-1)
        return (stack * weights).sum(dim=-1)

    def forward(
        self,
        const_one: torch.Tensor,
        x_vars: torch.Tensor,
        f_left_child: Optional[torch.Tensor],
        f_right_child: Optional[torch.Tensor],
        tau: float = 1.0,
        exp_order: int = 12,
        ln_order: int = 16,
        use_complex: bool = False,
    ) -> torch.Tensor:
        left_in = self.route(self.left_logits, const_one, x_vars, f_left_child, tau)
        right_in = self.route(self.right_logits, const_one, x_vars, f_right_child, tau)
        return eml(left_in, right_in, exp_order=exp_order, ln_order=ln_order, use_complex=use_complex)


class EMLTree(nn.Module):
    """Full binary EML tree of configurable depth (recursive, single-tree).

    Depth d has 2^d - 1 nodes and 2^d leaves.  Evaluation is bottom-up:
    all nodes at each level are computed in parallel across the batch.
    """

    def __init__(self, depth: int = 3, n_vars: int = 1):
        super().__init__()
        self.depth = depth
        self.n_vars = n_vars

        self.levels = nn.ModuleList()
        for level in range(depth):
            is_leaf = level == 0
            n_nodes = 2 ** (depth - 1 - level)
            self.levels.append(
                nn.ModuleList([EMLNode(is_leaf=is_leaf, n_vars=n_vars) for _ in range(n_nodes)])
            )

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        exp_order: int = 12,
        ln_order: int = 16,
        use_complex: bool = False,
    ) -> torch.Tensor:
        batch = x.shape[0]
        const_one = torch.ones(batch, device=x.device, dtype=x.dtype)
        x_vars = x.unsqueeze(-1) if x.dim() == 1 else x

        prev_outputs: list[torch.Tensor] | None = None
        for level_idx, level_nodes in enumerate(self.levels):
            current_outputs = []
            for node_idx, node in enumerate(level_nodes):
                f_left = None if level_idx == 0 else prev_outputs[2 * node_idx]
                f_right = None if level_idx == 0 else prev_outputs[2 * node_idx + 1]
                out = node(
                    const_one, x_vars, f_left, f_right,
                    tau=tau, exp_order=exp_order, ln_order=ln_order, use_complex=use_complex,
                )
                current_outputs.append(out)
            prev_outputs = current_outputs

        return prev_outputs[0]

    def snap_weights(self) -> dict[str, tuple[str, str]]:
        """Snap routing logits to one-hot argmax.  Returns route labels."""
        labels = ["1"] + [f"x{i}" for i in range(self.n_vars)] + ["f_child"]
        routes: dict[str, tuple[str, str]] = {}
        with torch.no_grad():
            for level_idx, level_nodes in enumerate(self.levels):
                for node_idx, node in enumerate(level_nodes):
                    for logits, side in [(node.left_logits, "L"), (node.right_logits, "R")]:
                        idx = logits.argmax().item()
                        logits.zero_()
                        logits[idx] = 10.0
                    li = min(node.left_logits.argmax().item(), len(labels) - 1)
                    ri = min(node.right_logits.argmax().item(), len(labels) - 1)
                    routes[f"L{level_idx}_N{node_idx}"] = (labels[li], labels[ri])
        return routes


class EMLTreeVectorized(nn.Module):
    """Fully vectorized multi-tree EML evaluation.

    Packs all N trees' routing logits into contiguous tensors and evaluates
    every node at the same depth in a single batched EML call.
    """

    def __init__(self, n_trees: int, depth: int, n_vars: int = 1):
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth
        self.n_vars = n_vars
        self.nodes_per_level = [2 ** (depth - 1 - l) for l in range(depth)]

        self.level_logits = nn.ParameterList()
        for level in range(depth):
            is_leaf = level == 0
            n_nodes = self.nodes_per_level[level]
            n_choices = 1 + n_vars + (0 if is_leaf else 1)
            self.level_logits.append(
                nn.Parameter(torch.randn(n_trees, n_nodes, n_choices, 2) * 0.1)
            )

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        exp_order: int = 12,
        ln_order: int = 16,
        use_complex: bool = False,
        clamp_input: float = 20.0,
    ) -> torch.Tensor:
        """Bottom-up evaluation.  x: [B, n_vars] -> returns [B, n_trees]."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        batch = x.shape[0]
        device, dtype = x.device, x.dtype
        const_one = torch.ones(batch, 1, 1, device=device, dtype=dtype)
        x_expanded = x.unsqueeze(1)  # [B, 1, n_vars]

        prev_outputs: torch.Tensor | None = None

        for level in range(self.depth):
            logits = self.level_logits[level]
            n_nodes = self.nodes_per_level[level]

            left_logits = logits[..., 0]
            right_logits = logits[..., 1]

            left_w = F.gumbel_softmax(
                left_logits.unsqueeze(0).expand(batch, -1, -1, -1).reshape(-1, left_logits.shape[-1]),
                tau=tau, hard=False,
            ).reshape(batch, self.n_trees, n_nodes, -1)

            right_w = F.gumbel_softmax(
                right_logits.unsqueeze(0).expand(batch, -1, -1, -1).reshape(-1, right_logits.shape[-1]),
                tau=tau, hard=False,
            ).reshape(batch, self.n_trees, n_nodes, -1)

            cands_l = [const_one.expand(batch, self.n_trees, n_nodes)]
            cands_r = [const_one.expand(batch, self.n_trees, n_nodes)]
            for v in range(self.n_vars):
                xv = x_expanded[:, :, v : v + 1].expand(batch, self.n_trees, n_nodes)
                cands_l.append(xv)
                cands_r.append(xv)

            if level > 0 and prev_outputs is not None:
                cands_l.append(prev_outputs[:, :, 0::2])
                cands_r.append(prev_outputs[:, :, 1::2])

            eml_x = (torch.stack(cands_l, dim=-1) * left_w).sum(dim=-1)
            eml_y = (torch.stack(cands_r, dim=-1) * right_w).sum(dim=-1)

            eml_x = eml_x.clamp(-clamp_input, clamp_input)
            if not use_complex:
                eml_y = eml_y.clamp(min=1e-30)

            prev_outputs = eml(eml_x, eml_y, exp_order=exp_order, ln_order=ln_order, use_complex=use_complex)

        return prev_outputs.squeeze(-1)
