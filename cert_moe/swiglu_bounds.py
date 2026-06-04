"""SwiGLU bound — Stage 2 verification for modern MoE experts.

Modern MoE models (Mixtral, OLMoE, DeepSeek) use SwiGLU [Shazeer 2020]
expert FFNs:

    E(h) = W_3 · ( SiLU(W_1 h) ⊙ (W_2 h) )

The bilinear term `SiLU(u) ⊙ v` is the technical core. We use:

  (a) Case-dispatched linear bounds on SiLU based on convexity:
        SiLU is convex on [u_-, u_+] ≈ [-2.4, 2.4], concave outside.
  (b) McCormick envelope on the bilinear φ(u)·v term.
  (c) Sign-aware substitution to obtain bounds linear in (u, v),
      composed with W_1, W_2 to bounds linear in h.

This module provides:
  - `silu_linear_bounds(u_lo, u_hi)` — sound per-coord linear bounds.
  - `swiglu_diff_bound(...)` — full bound on ||E_i - E_j||_∞ over a box.

Reference: framework derivation §a-1 (RESEARCH_OVERVIEW or memory).
"""
from __future__ import annotations

import torch


# SiLU inflection points (numerical), where 2 + u(1-2σ(u)) = 0.
# Computed once and hard-coded for speed; verified by numerical search.
U_INFL = 2.3994


# --------------------------------------------------------------------------- #
#  SiLU primitives                                                            #
# --------------------------------------------------------------------------- #

def silu(u: torch.Tensor) -> torch.Tensor:
    """SiLU(u) = u * sigmoid(u)."""
    return u * torch.sigmoid(u)


def silu_grad(u: torch.Tensor) -> torch.Tensor:
    """φ'(u) = σ(u) + u·σ(u)(1 - σ(u)) = σ(u)·(1 + u·(1 - σ(u)))."""
    s = torch.sigmoid(u)
    return s * (1 + u * (1 - s))


# --------------------------------------------------------------------------- #
#  Linear bounds on SiLU                                                      #
# --------------------------------------------------------------------------- #

def silu_interval(
    u_lo: torch.Tensor, u_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tight interval bound on SiLU over [u_lo, u_hi].

    SiLU has a unique minimum at u* ≈ -1.2785 with value ≈ -0.2785.
    Otherwise it's monotone.
    """
    U_STAR = -1.2785
    PHI_STAR = -0.2785

    phi_lo = silu(u_lo)
    phi_hi = silu(u_hi)

    # If interval contains u*, min is PHI_STAR; else min is at one endpoint
    contains_star = (u_lo <= U_STAR) & (u_hi >= U_STAR)
    phi_min_endpoints = torch.minimum(phi_lo, phi_hi)
    phi_lo_out = torch.where(
        contains_star, torch.full_like(u_lo, PHI_STAR), phi_min_endpoints
    )

    # Max is monotone for u ≥ u*, so always at one of the endpoints
    phi_hi_out = torch.maximum(phi_lo, phi_hi)
    return phi_lo_out, phi_hi_out


def silu_linear_bounds(
    u_lo: torch.Tensor, u_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-coordinate linear bounds: a_L·u + b_L ≤ SiLU(u) ≤ a_U·u + b_U.

    Case dispatch (per-coord):
      A: interval ⊆ [-U_INFL, U_INFL]   — fully convex
            chord       = upper bound
            tangent@mid = lower bound
      B: interval ⊆ [U_INFL, ∞)         — fully concave
            tangent@mid = upper bound
            chord       = lower bound
      C: interval ⊆ (-∞, -U_INFL]       — fully concave
            tangent@mid = upper bound
            chord       = lower bound
      D: crossing inflection — fall back to interval bounds
         (sound but loose; loses u-dependence in the bilinear step)

    Inputs and outputs are broadcastable tensors of identical shape.
    """
    eps = 1e-8

    phi_lo = silu(u_lo)
    phi_hi = silu(u_hi)
    safe_d = (u_hi - u_lo).clamp(min=eps)

    # Chord through endpoints
    chord_s = (phi_hi - phi_lo) / safe_d
    chord_b = phi_lo - chord_s * u_lo

    # Tangent at midpoint
    u_mid = 0.5 * (u_lo + u_hi)
    phi_mid = silu(u_mid)
    tan_s = silu_grad(u_mid)
    tan_b = phi_mid - tan_s * u_mid

    # Convex (Case A) when entire interval lies inside [-U_INFL, U_INFL]
    convex = (u_lo >= -U_INFL) & (u_hi <= U_INFL)
    # Fully concave (Cases B and C)
    concave_pos = u_lo >= U_INFL
    concave_neg = u_hi <= -U_INFL
    concave = concave_pos | concave_neg
    crossing = ~(convex | concave)

    # Initialize with chord/tangent (convex case as default)
    a_U = torch.where(convex, chord_s, tan_s)
    b_U = torch.where(convex, chord_b, tan_b)
    a_L = torch.where(convex, tan_s, chord_s)
    b_L = torch.where(convex, tan_b, chord_b)

    # Crossing case: use interval bound (zero slope, constant)
    if crossing.any():
        phi_int_lo, phi_int_hi = silu_interval(u_lo, u_hi)
        a_U = torch.where(crossing, torch.zeros_like(a_U), a_U)
        b_U = torch.where(crossing, phi_int_hi, b_U)
        a_L = torch.where(crossing, torch.zeros_like(a_L), a_L)
        b_L = torch.where(crossing, phi_int_lo, b_L)

    # NOTE: the per-case bound is sound only when we have the convexity
    # invariant. Sample-based audit at the endpoints (cheap):
    # We don't enforce here — the soundness is structural for Case A/B/C
    # and the crossing fallback is trivially sound (interval bound).

    return a_L, b_L, a_U, b_U


# --------------------------------------------------------------------------- #
#  Affine + zonotope range propagation                                        #
# --------------------------------------------------------------------------- #

def _affine_box(
    W: torch.Tensor, h_lo: torch.Tensor, h_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = W h, h ∈ [h_lo, h_hi] → [y_lo, y_hi]."""
    Wp = W.clamp(min=0)
    Wn = W.clamp(max=0)
    return Wp @ h_lo + Wn @ h_hi, Wp @ h_hi + Wn @ h_lo


def _affine_zonotope(
    W: torch.Tensor, center: torch.Tensor, V: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = W h, h = center + V^T diag(alpha) λ, λ ∈ [-1,1]^k."""
    y_c = W @ center
    y_gens = (W @ V.T) * alpha       # [out, k]
    rad = y_gens.abs().sum(-1)
    return y_c - rad, y_c + rad


# --------------------------------------------------------------------------- #
#  McCormick envelope on SiLU(u) * v                                          #
# --------------------------------------------------------------------------- #

def mccormick_silu_v_linear(
    u_lo: torch.Tensor, u_hi: torch.Tensor,
    v_lo: torch.Tensor, v_hi: torch.Tensor,
    a_L: torch.Tensor, b_L: torch.Tensor,
    a_U: torch.Tensor, b_U: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Linear bound on `SiLU(u) * v` in (u, v).

    Returns coefficients (alpha_u_L, alpha_v_L, gamma_L, alpha_u_U,
    alpha_v_U, gamma_U) so that:

        alpha_u_L·u + alpha_v_L·v + gamma_L
            ≤ SiLU(u)·v ≤
        alpha_u_U·u + alpha_v_U·v + gamma_U

    Constructed via McCormick envelopes with sign-aware substitution
    of the linear SiLU bound. We use the standard McCormick face

        xy ≥ x_lo·y + x·y_lo - x_lo·y_lo   (lower)
        xy ≤ x_hi·y + x·y_lo - x_hi·y_lo   (upper)

    with x = SiLU(u) ∈ [phi_lo, phi_hi] and y = v.
    """
    phi_lo, phi_hi = silu_interval(u_lo, u_hi)

    # ---- LOWER bound: x_lo·v + x·v_lo - x_lo·v_lo ----
    # Substitute x = SiLU(u). To LOWER BOUND the expression, the
    # coefficient of x is v_lo:
    #   - v_lo ≥ 0 ⇒ use lower bound of SiLU (a_L·u + b_L)
    #   - v_lo < 0 ⇒ use upper bound of SiLU (a_U·u + b_U), sign flips
    v_lo_pos = (v_lo >= 0).to(u_lo.dtype)
    v_lo_neg = 1 - v_lo_pos
    alpha_u_L = v_lo_pos * (a_L * v_lo) + v_lo_neg * (a_U * v_lo)
    alpha_v_L = phi_lo
    sub_b_L = v_lo_pos * b_L + v_lo_neg * b_U
    gamma_L = sub_b_L * v_lo - phi_lo * v_lo

    # ---- UPPER bound: x_hi·v + x·v_lo - x_hi·v_lo ----
    # (one McCormick face; another face is x_lo·v + x·v_hi - x_lo·v_hi.
    #  Element-wise min of the two gives the tighter upper envelope.)
    v_lo_pos_u = (v_lo >= 0).to(u_lo.dtype)
    v_lo_neg_u = 1 - v_lo_pos_u
    # For UPPER bound, coefficient of x is v_lo: same direction as
    # before but here we want the upper envelope, so:
    #   - v_lo ≥ 0 ⇒ use UPPER bound of SiLU
    #   - v_lo < 0 ⇒ use LOWER bound of SiLU (sign flips)
    alpha_u_U = v_lo_pos_u * (a_U * v_lo) + v_lo_neg_u * (a_L * v_lo)
    alpha_v_U = phi_hi
    sub_b_U = v_lo_pos_u * b_U + v_lo_neg_u * b_L
    gamma_U = sub_b_U * v_lo - phi_hi * v_lo

    return alpha_u_L, alpha_v_L, gamma_L, alpha_u_U, alpha_v_U, gamma_U


# --------------------------------------------------------------------------- #
#  Full SwiGLU expert bound                                                   #
# --------------------------------------------------------------------------- #

def swiglu_diff_bound_box(
    W1_i: torch.Tensor, W2_i: torch.Tensor, W3_i: torch.Tensor,
    W1_j: torch.Tensor, W2_j: torch.Tensor, W3_j: torch.Tensor,
    h_lo: torch.Tensor, h_hi: torch.Tensor,
) -> float:
    """Sound upper bound on ||E_i(h) - E_j(h)||_∞ over box [h_lo, h_hi].

    Each expert is E(h) = W_3 (SiLU(W_1 h) ⊙ (W_2 h)).
    The difference expert can be expressed as a single SwiGLU-like
    network with stacked weights and a sign flip on W_3 of j.
    """
    # Stack: u from W_1, v from W_2
    W1_stack = torch.cat([W1_i, W1_j], dim=0)         # [2 d_ff, d]
    W2_stack = torch.cat([W2_i, W2_j], dim=0)         # [2 d_ff, d]
    W3_concat = torch.cat([W3_i, -W3_j], dim=1)       # [d_out, 2 d_ff]

    # Pre-activation intervals
    u_lo, u_hi = _affine_box(W1_stack, h_lo, h_hi)
    v_lo, v_hi = _affine_box(W2_stack, h_lo, h_hi)

    # Linear bounds on SiLU per-coord
    a_L, b_L, a_U, b_U = silu_linear_bounds(u_lo, u_hi)

    # McCormick: SiLU(u)·v linear in (u, v) per-coord
    aL_u, aL_v, gL, aU_u, aU_v, gU = mccormick_silu_v_linear(
        u_lo, u_hi, v_lo, v_hi, a_L, b_L, a_U, b_U,
    )

    # Express in terms of h: u = W1_stack h, v = W2_stack h
    # Lower bound on D_o = Σ_k W3[o, k] · (aL_u[k] u_k + aL_v[k] v_k + gL[k])
    # Substitute u_k = W1[k] h, v_k = W2[k] h:
    #   D_o ≥ Σ_k W3[o, k] · (aL_u[k] (W1[k] · h) + aL_v[k] (W2[k] · h)
    #                          + gL[k])
    #       = ((W3 · diag(aL_u)) W1 + (W3 · diag(aL_v)) W2) · h
    #         + W3 · gL
    # But sign-aware: substitute lower bound if W3 > 0, else upper.
    W3 = W3_concat
    W3_pos = (W3 > 0).to(W3.dtype)
    W3_neg = 1 - W3_pos

    # For UPPER bound on D_o:
    coef_u_U = W3_pos * aU_u.unsqueeze(0) + W3_neg * aL_u.unsqueeze(0)
    coef_v_U = W3_pos * aU_v.unsqueeze(0) + W3_neg * aL_v.unsqueeze(0)
    coef_g_U = W3_pos * gU.unsqueeze(0) + W3_neg * gL.unsqueeze(0)
    A_U = (W3 * coef_u_U) @ W1_stack + (W3 * coef_v_U) @ W2_stack
    c_U = (W3 * coef_g_U).sum(-1)
    Up = A_U.clamp(min=0)
    Un = A_U.clamp(max=0)
    upper = Up @ h_hi + Un @ h_lo + c_U

    # For LOWER bound on D_o:
    coef_u_L = W3_pos * aL_u.unsqueeze(0) + W3_neg * aU_u.unsqueeze(0)
    coef_v_L = W3_pos * aL_v.unsqueeze(0) + W3_neg * aU_v.unsqueeze(0)
    coef_g_L = W3_pos * gL.unsqueeze(0) + W3_neg * gU.unsqueeze(0)
    A_L = (W3 * coef_u_L) @ W1_stack + (W3 * coef_v_L) @ W2_stack
    c_L = (W3 * coef_g_L).sum(-1)
    Ap = A_L.clamp(min=0)
    An = A_L.clamp(max=0)
    lower = Ap @ h_lo + An @ h_hi + c_L

    return torch.max(upper.abs().max(), lower.abs().max()).item()
