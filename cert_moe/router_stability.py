"""Stage 1 — Closed-form router stability verification.

Given router weight W_g ∈ R^{N×d} and input h, expert i is in TopK(h).
Under L_∞ perturbation ||δ||_∞ ≤ r, i remains in TopK iff for every
k currently NOT in TopK:
    ℓ_i(h) - ℓ_k(h) ≥ r · ||W_g[i] - W_g[k]||_1
(derivation: §1.2 of framework doc)

This module implements vectorized batch evaluation of this condition.
"""
from __future__ import annotations

import torch


def pairwise_router_l1(W_g: torch.Tensor) -> torch.Tensor:
    """L1 distance between rows of router weight: [N, N]."""
    # W_g: [N, d]
    diff = W_g.unsqueeze(0) - W_g.unsqueeze(1)  # [N, N, d]
    return diff.abs().sum(-1)                    # [N, N]


def stable_topk_mask(
    H: torch.Tensor,
    W_g: torch.Tensor,
    r: float,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each input h_b and each expert i, decide if i ∈ TopK(h_b) is
    stable under L_∞ perturbation of radius r.

    Args:
        H: [B, d] batch of hidden states
        W_g: [N, d] router weight
        r: perturbation radius
        K: top-K

    Returns:
        in_topk: [B, N] bool — true if i ∈ TopK(h_b)
        stable:  [B, N] bool — true if i ∈ TopK(h_b) AND stable
    """
    B, d = H.shape
    N = W_g.shape[0]

    logits = H @ W_g.T                                # [B, N]
    _, topk_idx = logits.topk(K, dim=-1)              # [B, K]

    in_topk = torch.zeros(B, N, dtype=torch.bool, device=H.device)
    in_topk.scatter_(1, topk_idx, True)

    # Margin matrix margin[b, i, k] = ℓ_i - ℓ_k
    margin = logits.unsqueeze(2) - logits.unsqueeze(1)        # [B, N, N]
    required = r * pairwise_router_l1(W_g).unsqueeze(0)       # [1, N, N]
    slack = margin - required                                  # [B, N, N]

    # We only require slack >= 0 for k ∉ TopK. Set k∈TopK positions to +inf.
    out_of_topk = ~in_topk                                     # [B, N]
    mask = out_of_topk.unsqueeze(1).expand(-1, N, -1)          # [B, N, N]
    masked_slack = torch.where(mask, slack, torch.full_like(slack, float("inf")))

    min_slack_per_i = masked_slack.min(dim=-1).values          # [B, N]
    stable = (min_slack_per_i >= 0) & in_topk
    return in_topk, stable


def stable_pair_mask(
    H: torch.Tensor,
    W_g: torch.Tensor,
    r: float,
    K: int,
) -> torch.Tensor:
    """For a candidate merge pair (i, j), find inputs h where {i, j}
    intersects TopK(h+δ) for all ||δ||_∞ ≤ r. This is the per-pair
    stable calibration set used to construct H*_{i,j}.

    Returns:
        pair_stable: [B, N, N] bool, pair_stable[b, i, j] = True if h_b
        is stable for the pair (i, j). Symmetric.
    """
    _, stable = stable_topk_mask(H, W_g, r, K)  # [B, N]
    # h_b is stable for pair (i, j) if at least one of {i, j} is stably in TopK.
    # We require the *stronger* condition: i or j individually is stable in TopK
    # (perturbation does not eject it). The "intersects TopK" condition would be
    # slightly weaker but harder to express in closed form.
    pair_stable = stable.unsqueeze(2) | stable.unsqueeze(1)   # [B, N, N]
    return pair_stable
