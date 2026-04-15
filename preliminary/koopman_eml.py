"""
koopman_eml.py – GPU-optimized EML Koopman Dictionary
======================================================

Uses Taylor-series approximations (Horner form) for exp() and ln()
so that the entire EML operator reduces to fused multiply-adds on GPU.
All tree levels are evaluated in parallel across the batch dimension.

EML(x, y) = exp(x) - ln(y)

References:
  Odrzywołek (2026), "All elementary functions from a single operator"
  arXiv:2603.21852v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================================
# 1. TAYLOR-SERIES PRIMITIVES (Horner form, GPU-vectorized)
# ============================================================================

def taylor_exp(x: torch.Tensor, order: int = 12) -> torch.Tensor:
    """
    exp(x) via Horner-form Taylor series with range reduction.
    
    Range reduction: exp(x) = 2^k * exp(r) where r = x - k*ln2, |r| <= ln2/2
    This keeps the Taylor argument small for fast convergence.
    
    Horner evaluation of exp(r) = 1 + r(1 + r/2(1 + r/3(1 + ...)))
    is N multiply-adds — pure FMA on GPU, no transcendentals.
    """
    ln2 = 0.6931471805599453
    
    # Range reduction: x = k*ln2 + r,  |r| <= ln2/2
    k = torch.round(x / ln2)
    r = x - k * ln2
    
    # Horner's method: evaluate from innermost term outward
    # exp(r) = sum_{n=0}^{order} r^n / n!
    # Horner form: 1 + r*(1 + r/2*(1 + r/3*(1 + ... r/order)))
    result = torch.ones_like(r)
    for n in range(order, 0, -1):
        result = 1.0 + (r / n) * result
    
    # Undo range reduction: exp(x) = exp(r) * 2^k
    return result * torch.pow(2.0, k)


def taylor_ln(y: torch.Tensor, order: int = 16) -> torch.Tensor:
    """
    ln(y) via range reduction + Horner-form series.
    
    Range reduction: y = m * 2^e  where m in [0.5, 1.0)
    Then ln(y) = e*ln2 + ln(m)
    
    For ln(m), use the identity:
      ln(m) = 2 * arctanh((m-1)/(m+1)) = 2 * sum_{k=0}^{N} u^(2k+1)/(2k+1)
    where u = (m-1)/(m+1).  This converges much faster than the naive
    ln(1+t) series because |u| <= 1/3 for m in [0.5, 1.0).
    
    Requires y > 0.  For complex extension, see taylor_ln_complex.
    """
    ln2 = 0.6931471805599453
    
    # frexp decomposition: y = m * 2^e,  m in [0.5, 1.0)
    # PyTorch doesn't have frexp, so we do it manually
    e = torch.floor(torch.log2(y.abs().clamp(min=1e-45)))
    m = y * torch.pow(2.0, -e)
    
    # Correct for edge cases in the decomposition
    mask = m < 0.5
    m = torch.where(mask, m * 2.0, m)
    e = torch.where(mask, e - 1.0, e)
    
    # arctanh series: u = (m-1)/(m+1), |u| <= 1/3
    u = (m - 1.0) / (m + 1.0)
    u2 = u * u
    
    # Horner-like accumulation for sum u^(2k+1)/(2k+1), k = N..0
    # = u * (1 + u2/3 * (1 + u2*3/(5) * (1 + ...)))
    # More directly: accumulate s = u^(2N+1)/(2N+1) + ... + u^3/3 + u
    terms = order // 2  # number of terms in arctanh series
    s = torch.zeros_like(u)
    for k in range(terms, -1, -1):
        coeff = 1.0 / (2 * k + 1)
        s = coeff + u2 * s  # Horner step
    s = u * s  # multiply by u once
    
    # ln(m) = 2 * arctanh(u) = 2 * s
    # ln(y) = e * ln2 + ln(m)
    return e * ln2 + 2.0 * s


def taylor_exp_complex(z: torch.Tensor, order: int = 12) -> torch.Tensor:
    """
    Complex exp(z) via Taylor series in Horner form.
    z is a complex tensor.  Range reduction on real part only.
    """
    ln2 = 0.6931471805599453
    
    # Range-reduce real part
    k = torch.round(z.real / ln2)
    r = torch.complex(z.real - k * ln2, z.imag)
    
    result = torch.ones_like(r)
    for n in range(order, 0, -1):
        result = 1.0 + (r / n) * result
    
    scale = torch.pow(2.0, k)
    return result * scale.to(z.dtype)


def taylor_ln_complex(w: torch.Tensor, order: int = 16) -> torch.Tensor:
    """
    Complex ln(w) = ln|w| + i*arg(w).
    Uses taylor_ln for the modulus, torch.angle for the argument.
    Principal branch: arg in (-pi, pi].
    """
    modulus = w.abs().clamp(min=1e-45)
    ln_mod = taylor_ln(modulus, order=order)
    angle = torch.angle(w)
    return torch.complex(ln_mod, angle)


# ============================================================================
# 2. VECTORIZED EML OPERATOR
# ============================================================================

class EMLOp(torch.autograd.Function):
    """
    Custom autograd for EML(x, y) = taylor_exp(x) - taylor_ln(y).
    
    Forward: pure polynomial (Horner) evaluation — no transcendentals.
    Backward: d/dx EML = exp(x),  d/dy EML = -1/y.
    These are also approximated via Taylor for consistency.
    """
    
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
        
        grad_x = grad_output * dfdx
        grad_y = grad_output * dfdy
        
        return grad_x, grad_y, None, None, None


def eml(x: torch.Tensor, y: torch.Tensor,
        exp_order: int = 12, ln_order: int = 16,
        use_complex: bool = False) -> torch.Tensor:
    """Vectorized EML(x,y) = exp(x) - ln(y) via Taylor approximations."""
    return EMLOp.apply(x, y, exp_order, ln_order, use_complex)


# ============================================================================
# 3. PARAMETERIZED EML TREE (Master Formula)
# ============================================================================

class EMLNode(nn.Module):
    """
    Single parameterized EML node.
    
    Each of the two inputs is routed via a Gumbel-softmax selector:
        input = softmax([α, β, γ]) · [1, x, f_child]
    
    During training, the softmax is soft (temperature τ > 0).
    At snap time, τ → 0 and weights discretize to one-hot.
    """
    
    def __init__(self, is_leaf: bool = False, n_vars: int = 1):
        super().__init__()
        self.is_leaf = is_leaf
        self.n_vars = n_vars
        
        # Routing logits for left and right inputs
        # Categories: [const_1, x_0, x_1, ..., x_{n-1}, f_child]
        n_choices = 1 + n_vars + (0 if is_leaf else 1)
        self.left_logits = nn.Parameter(torch.randn(n_choices) * 0.1)
        self.right_logits = nn.Parameter(torch.randn(n_choices) * 0.1)
    
    def route(self, logits: torch.Tensor, const_one: torch.Tensor,
              x_vars: torch.Tensor, f_child: Optional[torch.Tensor],
              tau: float) -> torch.Tensor:
        """
        Soft-select among {1, x_0, ..., x_{n-1}, f_child}.
        Uses Gumbel-softmax for differentiable discrete selection.
        """
        weights = F.gumbel_softmax(logits, tau=tau, hard=False)
        
        # Stack candidates: [batch, n_choices]
        candidates = [const_one]  # shape: [batch]
        for i in range(self.n_vars):
            if x_vars.dim() == 1:
                candidates.append(x_vars)
            else:
                candidates.append(x_vars[:, i])
        if f_child is not None:
            candidates.append(f_child)
        
        # [batch, n_choices]
        stack = torch.stack(candidates, dim=-1)
        # Weighted sum: [batch]
        return (stack * weights).sum(dim=-1)
    
    def forward(self, const_one: torch.Tensor, x_vars: torch.Tensor,
                f_left_child: Optional[torch.Tensor],
                f_right_child: Optional[torch.Tensor],
                tau: float = 1.0,
                exp_order: int = 12, ln_order: int = 16,
                use_complex: bool = False) -> torch.Tensor:
        """Compute EML(left_input, right_input) with soft routing."""
        left_in = self.route(self.left_logits, const_one, x_vars,
                             f_left_child, tau)
        right_in = self.route(self.right_logits, const_one, x_vars,
                              f_right_child, tau)
        
        return eml(left_in, right_in,
                    exp_order=exp_order, ln_order=ln_order,
                    use_complex=use_complex)


class EMLTree(nn.Module):
    """
    Full binary EML tree of configurable depth.
    
    Depth 1: single EML node (2 leaf inputs)           → 2 routing vectors
    Depth 2: 1 root + 2 children                       → 6 routing vectors (14 params for 1 var)
    Depth 3: 1 root + 2 children + 4 grandchildren     → 14 routing vectors
    Depth d: 2^d - 1 nodes total, 2^d leaves
    
    Evaluation proceeds bottom-up: all nodes at each level are computed
    in parallel across the batch.
    """
    
    def __init__(self, depth: int = 3, n_vars: int = 1):
        super().__init__()
        self.depth = depth
        self.n_vars = n_vars
        
        # Create nodes level by level (level 0 = leaves, level depth-1 = root)
        self.levels = nn.ModuleList()
        for level in range(depth):
            is_leaf = (level == 0)
            n_nodes = 2 ** (depth - 1 - level)
            level_nodes = nn.ModuleList([
                EMLNode(is_leaf=is_leaf, n_vars=n_vars)
                for _ in range(n_nodes)
            ])
            self.levels.append(level_nodes)
    
    def forward(self, x: torch.Tensor, tau: float = 1.0,
                exp_order: int = 12, ln_order: int = 16,
                use_complex: bool = False) -> torch.Tensor:
        """
        Bottom-up evaluation of the tree.
        x: [batch] or [batch, n_vars]
        Returns: [batch] — the tree output
        """
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype
        const_one = torch.ones(batch, device=device, dtype=dtype)
        
        if x.dim() == 1:
            x_vars = x.unsqueeze(-1)
        else:
            x_vars = x
        
        # Bottom-up: compute each level, passing results up
        prev_outputs = None
        
        for level_idx, level_nodes in enumerate(self.levels):
            current_outputs = []
            
            for node_idx, node in enumerate(level_nodes):
                if level_idx == 0:
                    # Leaf nodes: no child outputs
                    f_left = None
                    f_right = None
                else:
                    # Interior nodes: get outputs from children
                    f_left = prev_outputs[2 * node_idx]
                    f_right = prev_outputs[2 * node_idx + 1]
                
                out = node(const_one, x_vars, f_left, f_right,
                           tau=tau, exp_order=exp_order, ln_order=ln_order,
                           use_complex=use_complex)
                current_outputs.append(out)
            
            prev_outputs = current_outputs
        
        # Root is the single output at the top level
        return prev_outputs[0]
    
    def snap_weights(self) -> dict:
        """
        Snap all routing logits to one-hot (argmax).
        Returns a dict mapping node paths to their selected routes.
        """
        routes = {}
        labels = ['1'] + [f'x{i}' for i in range(self.n_vars)] + ['f_child']
        
        with torch.no_grad():
            for level_idx, level_nodes in enumerate(self.levels):
                for node_idx, node in enumerate(level_nodes):
                    left_idx = node.left_logits.argmax().item()
                    right_idx = node.right_logits.argmax().item()
                    
                    node.left_logits.zero_()
                    node.left_logits[left_idx] = 10.0  # hard one-hot
                    node.right_logits.zero_()
                    node.right_logits[right_idx] = 10.0
                    
                    n_l = min(left_idx, len(labels) - 1)
                    n_r = min(right_idx, len(labels) - 1)
                    routes[f'L{level_idx}_N{node_idx}'] = (labels[n_l], labels[n_r])
        
        return routes


# ============================================================================
# 4. VECTORIZED TREE EVALUATION (batched across all trees)
# ============================================================================

class EMLTreeVectorized(nn.Module):
    """
    Fully vectorized EML tree — all nodes at the same depth across all
    trees in the dictionary are computed in a single batched EML call.
    
    Instead of N separate EMLTree objects, this packs all routing logits
    into contiguous tensors for maximum GPU throughput.
    
    Shape convention:
        logits:  [n_trees, n_nodes_at_level, n_choices, 2]  (left/right)
        outputs: [batch, n_trees, n_nodes_at_level]
    """
    
    def __init__(self, n_trees: int, depth: int, n_vars: int = 1):
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth
        self.n_vars = n_vars
        
        # Pre-compute tree structure
        self.nodes_per_level = [2 ** (depth - 1 - l) for l in range(depth)]
        
        # Allocate logits for each level: [n_trees, n_nodes, n_choices, 2]
        # 2 = left input, right input
        self.level_logits = nn.ParameterList()
        for level in range(depth):
            is_leaf = (level == 0)
            n_nodes = self.nodes_per_level[level]
            n_choices = 1 + n_vars + (0 if is_leaf else 1)
            logits = nn.Parameter(torch.randn(n_trees, n_nodes, n_choices, 2) * 0.1)
            self.level_logits.append(logits)
    
    def forward(self, x: torch.Tensor, tau: float = 1.0,
                exp_order: int = 12, ln_order: int = 16,
                use_complex: bool = False,
                clamp_input: float = 20.0) -> torch.Tensor:
        """
        Batched bottom-up evaluation of all trees simultaneously.
        
        x: [batch, n_vars]  (or [batch] for univariate)
        Returns: [batch, n_trees]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # const_one: [batch, 1] for broadcasting
        const_one = torch.ones(batch, 1, 1, device=device, dtype=dtype)
        
        # x_vars: [batch, 1, n_vars] for broadcasting against [1, n_trees, n_nodes]
        x_expanded = x.unsqueeze(1)  # [batch, 1, n_vars]
        
        prev_outputs = None  # will be [batch, n_trees, n_nodes_at_prev_level]
        
        for level in range(self.depth):
            logits = self.level_logits[level]  # [n_trees, n_nodes, n_choices, 2]
            n_nodes = self.nodes_per_level[level]
            
            # Gumbel-softmax on the choice dimension
            # logits[..., 0] = left, logits[..., 1] = right
            left_logits = logits[..., 0]   # [n_trees, n_nodes, n_choices]
            right_logits = logits[..., 1]  # [n_trees, n_nodes, n_choices]
            
            # Expand for batch: [1, n_trees, n_nodes, n_choices]
            left_w = F.gumbel_softmax(
                left_logits.unsqueeze(0).expand(batch, -1, -1, -1)
                    .reshape(-1, left_logits.shape[-1]),
                tau=tau, hard=False
            ).reshape(batch, self.n_trees, n_nodes, -1)
            
            right_w = F.gumbel_softmax(
                right_logits.unsqueeze(0).expand(batch, -1, -1, -1)
                    .reshape(-1, right_logits.shape[-1]),
                tau=tau, hard=False
            ).reshape(batch, self.n_trees, n_nodes, -1)
            
            # Build candidate stack: [batch, n_trees, n_nodes, n_choices]
            # Candidates: [1, x_0, ..., x_{n-1}] for leaves
            #             [1, x_0, ..., x_{n-1}, f_child] for interior
            candidates_left = [const_one.expand(batch, self.n_trees, n_nodes)]
            candidates_right = [const_one.expand(batch, self.n_trees, n_nodes)]
            
            for v in range(self.n_vars):
                xv = x_expanded[:, :, v:v+1].expand(batch, self.n_trees, n_nodes)
                candidates_left.append(xv)
                candidates_right.append(xv)
            
            if level > 0 and prev_outputs is not None:
                # Each node i at this level gets children 2i and 2i+1
                # prev_outputs: [batch, n_trees, n_nodes_prev]
                left_children = prev_outputs[:, :, 0::2]    # even indices
                right_children = prev_outputs[:, :, 1::2]   # odd indices
                candidates_left.append(left_children)
                candidates_right.append(right_children)
            
            # Stack: [batch, n_trees, n_nodes, n_choices]
            stack_left = torch.stack(candidates_left, dim=-1)
            stack_right = torch.stack(candidates_right, dim=-1)
            
            # Weighted selection: [batch, n_trees, n_nodes]
            eml_x = (stack_left * left_w).sum(dim=-1)
            eml_y = (stack_right * right_w).sum(dim=-1)
            
            # Clamp to prevent overflow in nested exp
            eml_x = eml_x.clamp(-clamp_input, clamp_input)
            eml_y = eml_y.clamp(min=1e-30) if not use_complex else eml_y
            
            # Batched EML: [batch, n_trees, n_nodes]
            level_out = eml(eml_x, eml_y,
                            exp_order=exp_order, ln_order=ln_order,
                            use_complex=use_complex)
            
            prev_outputs = level_out
        
        # Root output: [batch, n_trees, 1] -> [batch, n_trees]
        return prev_outputs.squeeze(-1)


# ============================================================================
# 5. KOOPMAN EML MODEL
# ============================================================================

class KoopmanEML(nn.Module):
    """
    Full Koopman model with EML observable dictionary.
    
    Architecture:
        g(x) = [EML_tree_1(x), ..., EML_tree_N(x)]   (lifting)
        g(x_{k+1}) ≈ K @ g(x_k)                       (linear dynamics)
        x_hat = C @ g(x)                               (reconstruction)
    
    Loss:
        L = ||g(x_{k+1}) - K @ g(x_k)||^2             (prediction)
          + λ ||x_k - C @ g(x_k)||^2                   (reconstruction)
          + μ ||K||_F^2                                 (regularization)
    """
    
    def __init__(self, state_dim: int, n_observables: int = 32,
                 tree_depth: int = 3, exp_order: int = 12,
                 ln_order: int = 16, use_complex: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.n_obs = n_observables
        self.exp_order = exp_order
        self.ln_order = ln_order
        self.use_complex = use_complex
        
        # Observable dictionary: vectorized EML trees
        self.dictionary = EMLTreeVectorized(
            n_trees=n_observables,
            depth=tree_depth,
            n_vars=state_dim
        )
        
        # Koopman matrix K: [n_obs, n_obs]
        self.K = nn.Parameter(torch.eye(n_observables) +
                              torch.randn(n_observables, n_observables) * 0.01)
        
        # Decoder C: [state_dim, n_obs]
        self.C = nn.Linear(n_observables, state_dim, bias=False)
    
    def lift(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Lift state x to observable space: [batch, state_dim] -> [batch, n_obs]"""
        return self.dictionary(x, tau=tau,
                               exp_order=self.exp_order,
                               ln_order=self.ln_order,
                               use_complex=self.use_complex)
    
    def predict(self, g: torch.Tensor) -> torch.Tensor:
        """One-step prediction in observable space: g_{k+1} = K @ g_k"""
        return g @ self.K.T
    
    def reconstruct(self, g: torch.Tensor) -> torch.Tensor:
        """Decode from observable space back to state: x_hat = C @ g"""
        return self.C(g)
    
    def forward(self, x_k: torch.Tensor, x_k1: torch.Tensor,
                tau: float = 1.0) -> dict:
        """
        Full forward pass. Returns losses and intermediate values.
        
        x_k:  [batch, state_dim]  — current state
        x_k1: [batch, state_dim]  — next state
        """
        # Lift both states
        g_k = self.lift(x_k, tau=tau)
        g_k1 = self.lift(x_k1, tau=tau)
        
        # Predict and reconstruct
        g_k1_pred = self.predict(g_k)
        x_k_recon = self.reconstruct(g_k)
        
        # Losses
        pred_loss = F.mse_loss(g_k1_pred, g_k1)
        recon_loss = F.mse_loss(x_k_recon, x_k)
        reg_loss = torch.norm(self.K, p='fro') ** 2
        
        return {
            'pred_loss': pred_loss,
            'recon_loss': recon_loss,
            'reg_loss': reg_loss,
            'g_k': g_k,
            'g_k1': g_k1,
            'g_k1_pred': g_k1_pred,
            'x_k_recon': x_k_recon,
        }


# ============================================================================
# 6. TRAINING LOOP WITH ANNEALING
# ============================================================================

def train_koopman_eml(
    model: KoopmanEML,
    X_k: torch.Tensor,
    X_k1: torch.Tensor,
    n_epochs: int = 2000,
    lr: float = 1e-3,
    lambda_recon: float = 1.0,
    mu_reg: float = 1e-4,
    tau_start: float = 2.0,
    tau_end: float = 0.1,
    snap_threshold: float = 0.01,
    device: str = 'cuda',
    verbose: bool = True,
) -> dict:
    """
    Three-phase training:
      Phase 1 (60%): Soft training with annealing τ (exploration)
      Phase 2 (30%): Hardening with low τ (convergence to discrete)
      Phase 3 (10%): Snapped weights, fine-tune K and C only
    
    X_k, X_k1: [N, state_dim] consecutive state pairs from trajectory data
    """
    model = model.to(device)
    X_k = X_k.to(device)
    X_k1 = X_k1.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    history = {'total': [], 'pred': [], 'recon': [], 'tau': []}
    
    phase1_end = int(0.6 * n_epochs)
    phase2_end = int(0.9 * n_epochs)
    
    for epoch in range(n_epochs):
        # --- Temperature schedule ---
        if epoch < phase1_end:
            # Phase 1: anneal τ from tau_start to 1.0
            progress = epoch / phase1_end
            tau = tau_start * (1.0 / tau_start) ** progress
        elif epoch < phase2_end:
            # Phase 2: anneal τ from 1.0 to tau_end
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            tau = 1.0 * (tau_end / 1.0) ** progress
        else:
            # Phase 3: snapped, tau irrelevant (hard routing)
            tau = 0.01
            if epoch == phase2_end:
                # Snap weights and freeze dictionary
                if verbose:
                    print(f"\n[Epoch {epoch}] Snapping EML tree weights...")
                routes = {}
                for level_logits in model.dictionary.level_logits:
                    level_logits.requires_grad_(False)
                    # Hard snap to argmax
                    with torch.no_grad():
                        for t in range(level_logits.shape[0]):
                            for n in range(level_logits.shape[1]):
                                for side in range(2):
                                    idx = level_logits[t, n, :, side].argmax()
                                    level_logits[t, n, :, side] = -10.0
                                    level_logits[t, n, idx, side] = 10.0
                
                # Re-create optimizer for K and C only
                optimizer = torch.optim.Adam(
                    [model.K, *model.C.parameters()], lr=lr * 0.1
                )
        
        # --- Forward pass ---
        optimizer.zero_grad()
        out = model(X_k, X_k1, tau=tau)
        
        loss = (out['pred_loss']
                + lambda_recon * out['recon_loss']
                + mu_reg * out['reg_loss'])
        
        # --- Backward + step ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        # --- Logging ---
        history['total'].append(loss.item())
        history['pred'].append(out['pred_loss'].item())
        history['recon'].append(out['recon_loss'].item())
        history['tau'].append(tau)
        
        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            print(f"[{epoch:5d}/{n_epochs}]  τ={tau:.4f}  "
                  f"pred={out['pred_loss']:.6f}  "
                  f"recon={out['recon_loss']:.6f}  "
                  f"total={loss:.6f}")
    
    return history


# ============================================================================
# 7. KOOPMAN ANALYSIS UTILITIES
# ============================================================================

def koopman_eigendecomposition(model: KoopmanEML) -> dict:
    """
    Extract Koopman eigenvalues, eigenvectors, and modes.
    
    Returns:
        eigenvalues:  [n_obs] complex — Koopman eigenvalues
        eigenvectors: [n_obs, n_obs] complex — right eigenvectors
        modes:        [state_dim, n_obs] complex — Koopman modes (C @ V)
        frequencies:  [n_obs] real — oscillation frequencies (Im(ln(λ))/(2π))
        growth_rates: [n_obs] real — growth/decay rates (Re(ln(λ)))
    """
    K = model.K.detach().cpu().numpy()
    import numpy as np
    
    eigenvalues, eigenvectors = np.linalg.eig(K)
    
    C = model.C.weight.detach().cpu().numpy()  # [state_dim, n_obs]
    modes = C @ eigenvectors
    
    # Extract dynamics from eigenvalues
    ln_eig = np.log(eigenvalues.astype(complex) + 1e-30)
    frequencies = ln_eig.imag / (2 * np.pi)
    growth_rates = ln_eig.real
    
    # Sort by magnitude (most dominant first)
    order = np.argsort(-np.abs(eigenvalues))
    
    return {
        'eigenvalues': eigenvalues[order],
        'eigenvectors': eigenvectors[:, order],
        'modes': modes[:, order],
        'frequencies': frequencies[order],
        'growth_rates': growth_rates[order],
    }


def extract_eml_formulas(model: KoopmanEML) -> list:
    """
    After snapping, read out the symbolic EML formula for each observable.
    Returns a list of nested string expressions.
    """
    formulas = []
    dict_mod = model.dictionary
    n_vars = dict_mod.n_vars
    
    var_names = [f'x{i}' for i in range(n_vars)]
    
    for tree_idx in range(dict_mod.n_trees):
        # Reconstruct formula bottom-up
        prev_exprs = None
        
        for level in range(dict_mod.depth):
            logits = dict_mod.level_logits[level]  # [n_trees, n_nodes, n_choices, 2]
            n_nodes = dict_mod.nodes_per_level[level]
            
            current_exprs = []
            for node_idx in range(n_nodes):
                left_idx = logits[tree_idx, node_idx, :, 0].argmax().item()
                right_idx = logits[tree_idx, node_idx, :, 1].argmax().item()
                
                labels = ['1'] + var_names
                if level > 0:
                    labels.append(None)  # placeholder for child
                
                def resolve(idx, child_left, child_right):
                    if idx == 0:
                        return '1'
                    elif idx <= n_vars:
                        return var_names[idx - 1]
                    else:
                        # f_child — but which child? left gets even, right gets odd
                        # node_idx's left child is 2*node_idx, right is 2*node_idx+1
                        return None  # resolved below
                
                left_expr = resolve(left_idx, None, None)
                right_expr = resolve(right_idx, None, None)
                
                if left_expr is None and prev_exprs is not None:
                    left_expr = prev_exprs[2 * node_idx]
                if right_expr is None and prev_exprs is not None:
                    right_expr = prev_exprs[2 * node_idx + 1]
                
                if left_expr is None:
                    left_expr = '?'
                if right_expr is None:
                    right_expr = '?'
                
                current_exprs.append(f'eml({left_expr}, {right_expr})')
            
            prev_exprs = current_exprs
        
        formulas.append(prev_exprs[0])
    
    return formulas


# ============================================================================
# 8. DEMO: DUFFING OSCILLATOR
# ============================================================================

def demo_duffing():
    """
    Demo: learn Koopman observables for the Duffing oscillator
        dx/dt = y
        dy/dt = x - x^3 - 0.3*y
    
    Known Koopman eigenfunctions include polynomials and 
    exponential-polynomial products — all elementary, so EML can find them.
    """
    import numpy as np
    
    print("=" * 70)
    print("Koopman-EML Demo: Duffing Oscillator")
    print("=" * 70)
    
    # Generate trajectory data
    dt = 0.01
    n_steps = 5000
    delta = 0.3
    
    def duffing_step(state, dt):
        x, y = state
        dx = y
        dy = x - x**3 - delta * y
        return np.array([x + dx * dt, y + dy * dt])
    
    # Multiple trajectories from random ICs
    n_traj = 20
    states = []
    for _ in range(n_traj):
        s = np.random.randn(2) * 0.5
        traj = [s]
        for _ in range(n_steps):
            s = duffing_step(s, dt)
            traj.append(s)
        states.append(np.array(traj))
    
    states = np.concatenate(states, axis=0)
    X_k = torch.tensor(states[:-1], dtype=torch.float32)
    X_k1 = torch.tensor(states[1:], dtype=torch.float32)
    
    # Subsample for speed
    idx = torch.randperm(len(X_k))[:10000]
    X_k, X_k1 = X_k[idx], X_k1[idx]
    
    print(f"Training data: {X_k.shape[0]} state pairs, dim={X_k.shape[1]}")
    
    # Build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = KoopmanEML(
        state_dim=2,
        n_observables=16,
        tree_depth=3,
        exp_order=10,
        ln_order=12,
        use_complex=False,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"EML tree params:  {sum(p.numel() for p in model.dictionary.parameters()):,}")
    print()
    
    # Train
    history = train_koopman_eml(
        model, X_k, X_k1,
        n_epochs=1000,
        lr=3e-3,
        lambda_recon=0.5,
        mu_reg=1e-4,
        tau_start=2.0,
        tau_end=0.1,
        device=device,
        verbose=True,
    )
    
    # Extract results
    print("\n" + "=" * 70)
    print("Koopman Eigendecomposition")
    print("=" * 70)
    eig = koopman_eigendecomposition(model)
    for i in range(min(8, len(eig['eigenvalues']))):
        lam = eig['eigenvalues'][i]
        print(f"  λ_{i} = {lam:.4f}  |λ|={abs(lam):.4f}  "
              f"freq={eig['frequencies'][i]:.4f}  "
              f"growth={eig['growth_rates'][i]:.4f}")
    
    print("\n" + "=" * 70)
    print("Discovered EML Observables (snapped formulas)")
    print("=" * 70)
    formulas = extract_eml_formulas(model)
    for i, f in enumerate(formulas):
        print(f"  g_{i}(x) = {f}")
    
    return model, history


if __name__ == '__main__':
    demo_duffing()
