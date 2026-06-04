"""Stage 2 — Sound upper bound on ||E_i(h) - E_j(h)||_∞ over input box.

Expert form (toy MoE): E(h) = W2 @ ReLU(W1 @ h).

Difference network: D(h) = E_i(h) - E_j(h)
                         = W2_i @ ReLU(W1_i @ h) - W2_j @ ReLU(W1_j @ h)

We express D as a single ReLU network with stacked weights:
    u = [W1_i; W1_j] @ h ∈ R^{2 d_ff}
    z = ReLU(u)
    D = [W2_i, -W2_j] @ z ∈ R^{d}

This lets us apply standard IBP and a simple backward CROWN bound
uniformly. For the prototype we implement both:

  - IBP (Interval Bound Propagation): O(d·d_ff) per pair, loose.
  - Backward CROWN with adaptive ReLU relaxation: tighter, modest cost.

Both produce a sound upper bound on max_h ||D(h)||_∞ over a box
input region [h_lo, h_hi].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Affine interval propagation                                                #
# --------------------------------------------------------------------------- #

def affine_interval(
    W: torch.Tensor, h_lo: torch.Tensor, h_hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For y = W @ h, return [y_lo, y_hi] given h ∈ [h_lo, h_hi].

    Standard interval arithmetic:
        y_lo = W^+ @ h_lo + W^- @ h_hi
        y_hi = W^+ @ h_hi + W^- @ h_lo
    where W^+ = max(W, 0), W^- = min(W, 0).
    """
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    y_lo = W_pos @ h_lo + W_neg @ h_hi
    y_hi = W_pos @ h_hi + W_neg @ h_lo
    return y_lo, y_hi


# --------------------------------------------------------------------------- #
#  IBP bound on difference network                                            #
# --------------------------------------------------------------------------- #

def ibp_diff_bound(
    W1_i: torch.Tensor, W2_i: torch.Tensor,
    W1_j: torch.Tensor, W2_j: torch.Tensor,
    h_lo: torch.Tensor, h_hi: torch.Tensor,
) -> float:
    """Interval Bound Propagation on the stacked difference network.

    Returns: sound upper bound on max_{h ∈ box} ||E_i(h) - E_j(h)||_∞
    """
    W1_stack = torch.cat([W1_i, W1_j], dim=0)                # [2 d_ff, d]
    W2_concat = torch.cat([W2_i, -W2_j], dim=1)              # [d, 2 d_ff]

    # Affine layer 1
    u_lo, u_hi = affine_interval(W1_stack, h_lo, h_hi)        # [2 d_ff]
    # ReLU
    z_lo = u_lo.clamp(min=0)
    z_hi = u_hi.clamp(min=0)
    # Affine layer 2
    y_lo, y_hi = affine_interval(W2_concat, z_lo, z_hi)       # [d]

    # ||D||_∞ ≤ max(|y_lo|, |y_hi|) componentwise, then max over dims
    return torch.max(y_lo.abs().max(), y_hi.abs().max()).item()


# --------------------------------------------------------------------------- #
#  Backward CROWN with ReLU adaptive relaxation                                #
# --------------------------------------------------------------------------- #

def _relu_relaxation_coefs(
    u_lo: torch.Tensor, u_hi: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each pre-activation neuron, return linear lower/upper bounds:

        a_L · u + b_L  ≤  ReLU(u)  ≤  a_U · u + b_U

    valid for u ∈ [u_lo, u_hi]. Three cases:

      Inactive (u_hi ≤ 0):  ReLU(u) = 0 → a = 0, b = 0 for both.
      Active   (u_lo ≥ 0):  ReLU(u) = u → a = 1, b = 0 for both.
      Crossing (u_lo < 0 < u_hi):
        Upper bound = chord: slope = u_hi/(u_hi - u_lo), intercept = -slope · u_lo
        Lower bound = α · u  with α ∈ [0, 1]
          - alpha=None: CROWN-Ada heuristic (slope = 1 if u_hi ≥ |u_lo|, else 0)
          - alpha provided: learnable per-neuron slope (α-CROWN).
    """
    a_L = torch.zeros_like(u_lo)
    b_L = torch.zeros_like(u_lo)
    a_U = torch.zeros_like(u_lo)
    b_U = torch.zeros_like(u_lo)

    active = u_lo >= 0
    a_L[active] = 1.0
    a_U[active] = 1.0

    crossing = (u_lo < 0) & (u_hi > 0)
    # Upper chord
    slope = u_hi[crossing] / (u_hi[crossing] - u_lo[crossing])
    a_U[crossing] = slope
    b_U[crossing] = -slope * u_lo[crossing]
    # Lower slope (heuristic or α-CROWN)
    if alpha is None:
        lower_slope = (u_hi[crossing] >= -u_lo[crossing]).float()
    else:
        # alpha is shape [2 d_ff], values in [0, 1] for crossing neurons
        # (clamped by sigmoid in the optimizer wrapper)
        lower_slope = alpha[crossing].clamp(0.0, 1.0)
    a_L[crossing] = lower_slope
    # b_L stays 0

    # Inactive case: all zeros (already initialized)
    return a_L, b_L, a_U, b_U


def _crown_bound_inner(
    W1_stack: torch.Tensor,
    W2_concat: torch.Tensor,
    h_lo: torch.Tensor,
    h_hi: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable inner bound computation (for α-CROWN optimization).

    Returns a SCALAR (max over output dims of |upper|, |lower|).
    Keeps gradients flowing wrt `alpha`.
    """
    u_lo, u_hi = affine_interval(W1_stack, h_lo, h_hi)
    a_L, b_L, a_U, b_U = _relu_relaxation_coefs(u_lo, u_hi, alpha=alpha)

    W2 = W2_concat
    W2_pos_mask = (W2 > 0).float()
    W2_neg_mask = (W2 < 0).float()

    # Upper bound on D_o
    coef_a = W2_pos_mask * a_U.unsqueeze(0) + W2_neg_mask * a_L.unsqueeze(0)
    coef_b = W2_pos_mask * b_U.unsqueeze(0) + W2_neg_mask * b_L.unsqueeze(0)
    A_up = (W2 * coef_a) @ W1_stack                # [d_out, d]
    c_up = (W2 * coef_b).sum(-1)
    upper = A_up.clamp(min=0) @ h_hi + A_up.clamp(max=0) @ h_lo + c_up

    # Lower bound on D_o
    coef_a_lo = W2_pos_mask * a_L.unsqueeze(0) + W2_neg_mask * a_U.unsqueeze(0)
    coef_b_lo = W2_pos_mask * b_L.unsqueeze(0) + W2_neg_mask * b_U.unsqueeze(0)
    A_lo = (W2 * coef_a_lo) @ W1_stack
    c_lo = (W2 * coef_b_lo).sum(-1)
    lower = A_lo.clamp(min=0) @ h_lo + A_lo.clamp(max=0) @ h_hi + c_lo

    return torch.max(upper.abs().max(), lower.abs().max())


def alpha_crown_diff_bound(
    W1_i: torch.Tensor, W2_i: torch.Tensor,
    W1_j: torch.Tensor, W2_j: torch.Tensor,
    h_lo: torch.Tensor, h_hi: torch.Tensor,
    n_iters: int = 30,
    lr: float = 0.5,
) -> float:
    """α-CROWN: optimize per-neuron lower-bound slopes to minimize the bound.

    For each crossing ReLU, parameterize α via sigmoid(θ) ∈ (0, 1).
    Gradient descent on the bound (a scalar) wrt θ.
    """
    W1_stack = torch.cat([W1_i, W1_j], dim=0)
    W2_concat = torch.cat([W2_i, -W2_j], dim=1)

    # Initialize α at the heuristic choice
    u_lo, u_hi = affine_interval(W1_stack, h_lo, h_hi)
    init = torch.where(u_hi >= -u_lo,
                       torch.full_like(u_lo, 3.0),    # sigmoid(3) ≈ 0.95
                       torch.full_like(u_lo, -3.0))   # sigmoid(-3) ≈ 0.05
    theta = init.clone().requires_grad_(True)

    opt = torch.optim.Adam([theta], lr=lr)
    best = float("inf")
    for _ in range(n_iters):
        opt.zero_grad()
        alpha = torch.sigmoid(theta)
        bound = _crown_bound_inner(W1_stack, W2_concat, h_lo, h_hi, alpha=alpha)
        bound.backward()
        opt.step()
        best = min(best, bound.item())
    return best


def crown_diff_bound(
    W1_i: torch.Tensor, W2_i: torch.Tensor,
    W1_j: torch.Tensor, W2_j: torch.Tensor,
    h_lo: torch.Tensor, h_hi: torch.Tensor,
) -> float:
    """Backward CROWN bound on the stacked difference network.

    For each output dimension o, computes upper bound on D_o(h) and
    lower bound on D_o(h) over box; returns max |D_o| upper bound.
    """
    W1_stack = torch.cat([W1_i, W1_j], dim=0)        # [2 d_ff, d]
    W2_concat = torch.cat([W2_i, -W2_j], dim=1)      # [d, 2 d_ff]
    d_out = W2_concat.shape[0]

    # Pre-activation interval (for ReLU relaxation)
    u_lo, u_hi = affine_interval(W1_stack, h_lo, h_hi)        # [2 d_ff]
    a_L, b_L, a_U, b_U = _relu_relaxation_coefs(u_lo, u_hi)

    # We want upper bound on D_o = sum_k W2_concat[o, k] · ReLU(u_k)
    # Substitute ReLU(u_k) by its linear bound based on sign of W2_concat[o, k]:
    #   if W2_concat[o, k] > 0: ReLU(u_k) ≤ a_U[k] u_k + b_U[k]
    #   if W2_concat[o, k] < 0: ReLU(u_k) ≥ a_L[k] u_k + b_L[k] (need lower bound, sign flips)
    # Final form: D_o ≤ sum_k W2_concat[o, k] · (chosen_a · u_k + chosen_b)
    #
    # Then u_k = W1_stack[k] @ h, so D_o ≤ (linear in h) → maximize over box.

    # Build coefficients per output dim
    # Shape strategy: do it for all o at once with masking
    W2 = W2_concat                                    # [d_out, 2 d_ff]
    W2_pos_mask = (W2 > 0).float()
    W2_neg_mask = (W2 < 0).float()

    # For upper bound on D_o, use a_U where W2 > 0, else a_L
    coef_a = W2_pos_mask * a_U.unsqueeze(0) + W2_neg_mask * a_L.unsqueeze(0)  # [d_out, 2 d_ff]
    coef_b = W2_pos_mask * b_U.unsqueeze(0) + W2_neg_mask * b_L.unsqueeze(0)  # [d_out, 2 d_ff]

    # Linear part in h: A_h[o, :] = sum_k W2[o, k] · coef_a[o, k] · W1_stack[k, :]
    # Equivalently: A_h = (W2 * coef_a) @ W1_stack
    A_h_upper = (W2 * coef_a) @ W1_stack              # [d_out, d]
    # Constant part: c[o] = sum_k W2[o, k] · coef_b[o, k]
    c_upper = (W2 * coef_b).sum(-1)                    # [d_out]

    # Maximize A_h_upper @ h + c_upper over h ∈ box
    # max h: positive entries take h_hi, negative take h_lo
    A_pos = A_h_upper.clamp(min=0)
    A_neg = A_h_upper.clamp(max=0)
    upper = A_pos @ h_hi + A_neg @ h_lo + c_upper       # [d_out]

    # Symmetric: lower bound (swap roles of a_L/a_U, b_L/b_U)
    coef_a_lo = W2_pos_mask * a_L.unsqueeze(0) + W2_neg_mask * a_U.unsqueeze(0)
    coef_b_lo = W2_pos_mask * b_L.unsqueeze(0) + W2_neg_mask * b_U.unsqueeze(0)
    A_h_lower = (W2 * coef_a_lo) @ W1_stack
    c_lower = (W2 * coef_b_lo).sum(-1)
    A_pos = A_h_lower.clamp(min=0)
    A_neg = A_h_lower.clamp(max=0)
    lower = A_pos @ h_lo + A_neg @ h_hi + c_lower

    # ||D||_∞ over box ≤ max(|upper|, |lower|) per dim, then max over dims
    return torch.max(upper.abs().max(), lower.abs().max()).item()


# --------------------------------------------------------------------------- #
#  Convenience wrapper                                                         #
# --------------------------------------------------------------------------- #

def pair_diff_bound(
    expert_i, expert_j,
    h_lo: torch.Tensor, h_hi: torch.Tensor,
    method: str = "crown",
    alpha_iters: int = 30,
) -> float:
    """Compute sound upper bound on ||E_i - E_j||_∞ over [h_lo, h_hi]."""
    W1_i, W2_i = expert_i.W1.detach(), expert_i.W2.detach()
    W1_j, W2_j = expert_j.W1.detach(), expert_j.W2.detach()
    if method == "ibp":
        return ibp_diff_bound(W1_i, W2_i, W1_j, W2_j, h_lo, h_hi)
    elif method == "crown":
        return crown_diff_bound(W1_i, W2_i, W1_j, W2_j, h_lo, h_hi)
    elif method == "alpha_crown":
        return alpha_crown_diff_bound(W1_i, W2_i, W1_j, W2_j,
                                      h_lo, h_hi, n_iters=alpha_iters)
    else:
        raise ValueError(f"Unknown method: {method}")


# --------------------------------------------------------------------------- #
#  Zonotope input region                                                       #
# --------------------------------------------------------------------------- #

def zonotope_from_data(
    H: torch.Tensor, n_components: int | None = None,
    margin: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a zonotope outer-approximating calibration data H.

    Form:  z = c + V^T diag(α) λ,   λ ∈ [-1, 1]^k
    where:
        c  = mean(H)                         center
        V  = top-k PCA components of (H-c)   [k, d]  (rows are generators)
        α  = max-abs projection of (H-c) onto each component, * (1+margin)

    Returns (center, V, alpha) with shapes ([d], [k, d], [k]).
    """
    center = H.mean(0)
    H_c = H - center
    # SVD: H_c = U S Vt
    _, _, Vt = torch.linalg.svd(H_c, full_matrices=False)
    if n_components is None:
        n_components = min(H.shape[0] - 1, H.shape[1])
    V = Vt[:n_components]                              # [k, d]
    proj = H_c @ V.T                                    # [B, k]
    alpha = proj.abs().max(0).values * (1.0 + margin)   # [k]
    return center, V, alpha


def _affine_zonotope_interval(
    W: torch.Tensor, center: torch.Tensor, V: torch.Tensor, alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For y = W h, h ∈ zonotope(center, V, alpha), return [y_lo, y_hi].

    y = W center + W V^T diag(alpha) λ, λ ∈ [-1,1]^k
    y_lo[o] = (W center)[o] - |(W V^T)[o, :] * alpha|.sum()
    y_hi[o] = (W center)[o] + same
    """
    y_c = W @ center                       # [out]
    y_gens = (W @ V.T) * alpha             # [out, k]
    rad = y_gens.abs().sum(-1)              # [out]
    return y_c - rad, y_c + rad


def crown_diff_bound_zonotope(
    W1_i: torch.Tensor, W2_i: torch.Tensor,
    W1_j: torch.Tensor, W2_j: torch.Tensor,
    center: torch.Tensor, V: torch.Tensor, alpha_coef: torch.Tensor,
    relu_alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward CROWN over a zonotope input.

    Returns a scalar (max |upper|, |lower| over output dims).
    Keeps gradients flowing wrt `relu_alpha` for α-CROWN optimization.
    """
    W1_stack = torch.cat([W1_i, W1_j], dim=0)
    W2_concat = torch.cat([W2_i, -W2_j], dim=1)

    u_lo, u_hi = _affine_zonotope_interval(W1_stack, center, V, alpha_coef)
    a_L, b_L, a_U, b_U = _relu_relaxation_coefs(u_lo, u_hi, alpha=relu_alpha)

    W2_pos = (W2_concat > 0).float()
    W2_neg = (W2_concat < 0).float()

    # Upper bound coefficients
    coef_a = W2_pos * a_U.unsqueeze(0) + W2_neg * a_L.unsqueeze(0)
    coef_b = W2_pos * b_U.unsqueeze(0) + W2_neg * b_L.unsqueeze(0)
    A_up = (W2_concat * coef_a) @ W1_stack          # [d_out, d]
    c_up = (W2_concat * coef_b).sum(-1)              # [d_out]
    # Max over zonotope: A center + |A V^T| diag(alpha) summed
    A_up_gens = (A_up @ V.T) * alpha_coef            # [d_out, k]
    upper = A_up @ center + A_up_gens.abs().sum(-1) + c_up

    # Lower bound coefficients
    coef_a_lo = W2_pos * a_L.unsqueeze(0) + W2_neg * a_U.unsqueeze(0)
    coef_b_lo = W2_pos * b_L.unsqueeze(0) + W2_neg * b_U.unsqueeze(0)
    A_lo = (W2_concat * coef_a_lo) @ W1_stack
    c_lo = (W2_concat * coef_b_lo).sum(-1)
    A_lo_gens = (A_lo @ V.T) * alpha_coef
    lower = A_lo @ center - A_lo_gens.abs().sum(-1) + c_lo

    return torch.max(upper.abs().max(), lower.abs().max())


def alpha_crown_zonotope_bound(
    expert_i, expert_j,
    center: torch.Tensor, V: torch.Tensor, alpha_coef: torch.Tensor,
    n_iters: int = 30, lr: float = 0.5,
) -> float:
    """α-CROWN bound over zonotope input."""
    W1_i = expert_i.W1.detach()
    W2_i = expert_i.W2.detach()
    W1_j = expert_j.W1.detach()
    W2_j = expert_j.W2.detach()

    W1_stack = torch.cat([W1_i, W1_j], dim=0)
    u_lo, u_hi = _affine_zonotope_interval(W1_stack, center, V, alpha_coef)
    init = torch.where(u_hi >= -u_lo,
                       torch.full_like(u_lo, 3.0),
                       torch.full_like(u_lo, -3.0))
    theta = init.clone().requires_grad_(True)

    opt = torch.optim.Adam([theta], lr=lr)
    best = float("inf")
    for _ in range(n_iters):
        opt.zero_grad()
        relu_a = torch.sigmoid(theta)
        b = crown_diff_bound_zonotope(
            W1_i, W2_i, W1_j, W2_j,
            center, V, alpha_coef, relu_alpha=relu_a,
        )
        b.backward()
        opt.step()
        best = min(best, b.item())
    return best


def empirical_pair_diff(
    expert_i, expert_j, H: torch.Tensor
) -> float:
    """For sanity check: empirical max ||E_i(h) - E_j(h)||_∞ on samples H."""
    with torch.no_grad():
        d_i = expert_i(H) - expert_j(H)
    return d_i.abs().max().item()
