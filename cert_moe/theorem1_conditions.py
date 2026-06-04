"""Closed-form verification of Theorem 1's three sufficient conditions
for the post-merge input-conditional bound:

  (Stab) h ∈ ⋂_{i ∈ TopK(h)} H*_{C(i)}                — Stage 1 (router stability)
  (MC)   ∀C ∈ Π : |TopK(h) ∩ C| ≤ 1                   — at most one cluster member in top-K
  (TS)   min_{C ∈ expected_topK} ℓ̂_C > max_{C ∉ expected} ℓ̂_C
                                                       — merged router preserves top-K composition
  (EM)   Ξ(h) := Σ_{C ∈ expected_topK} Σ_{k in C minus TopK(h)} p_k(h) ≤ δ
                                                       — cluster's non-top-K members carry little mass

All four checks operate on the linear router only — expert FFNs are
not invoked. Per Theorem 1, satisfaction of these gives:
    ||M(h) - M̂(h)||_∞  ≤  ε + (2 δ / S(h)) · E_max(h)
where ε is the Stage 2 verified pairwise bound used in clustering.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConditionMasks:
    """Per-token results of Theorem 1 condition checks.

    Each mask is shape [B], True iff the corresponding condition holds.
    """
    mc: torch.Tensor   # Merge-Compatibility
    ts: torch.Tensor   # Top-K Stability (under merge, not under perturbation)
    em: torch.Tensor   # Extra Mass bounded
    xi: torch.Tensor   # Actual Ξ(h) values, for analysis

    @property
    def all_certified(self) -> torch.Tensor:
        """Tokens satisfying every condition (Theorem 1 applies)."""
        return self.mc & self.ts & self.em


def _cluster_member_lists(plan) -> list[list[int]]:
    """For each cluster id c, list of expert indices belonging to it."""
    return [list(c) for c in plan.clusters]


def _merged_logits(
    logits: torch.Tensor,
    plan,
) -> torch.Tensor:
    """Log-sum-exp-merge router logits per cluster.

    Args:
        logits: [B, N]
        plan:   MergePlan
    Returns:
        merged: [B, n_clusters]
    """
    B = logits.shape[0]
    n_c = len(plan.clusters)
    merged = torch.full((B, n_c), float("-inf"), device=logits.device)
    for c, members in enumerate(plan.clusters):
        idx = torch.as_tensor(members, dtype=torch.long, device=logits.device)
        merged[:, c] = torch.logsumexp(logits.index_select(1, idx), dim=-1)
    return merged


def check_conditions(
    H: torch.Tensor,
    W_g: torch.Tensor,
    plan,
    K: int,
    delta_threshold: float,
) -> ConditionMasks:
    """Closed-form per-token check of (MC), (TS), (EM)."""
    B = H.shape[0]
    N = W_g.shape[0]

    logits = H @ W_g.T                                   # [B, N]
    probs = logits.softmax(-1)
    topk_vals, topk_idx = logits.topk(K, dim=-1)         # [B, K]

    label_of = torch.as_tensor(plan.label_of, dtype=torch.long,
                               device=H.device)          # [N]

    # ---------- (MC) ----------
    # For each token, count how many of its top-K experts share a cluster.
    topk_clusters = label_of[topk_idx]                   # [B, K]
    sorted_tc, _ = topk_clusters.sort(dim=-1)
    duplicate = (sorted_tc[:, 1:] == sorted_tc[:, :-1]).any(dim=-1)  # [B]
    mc = ~duplicate                                       # [B]

    # ---------- (TS) ----------
    # Merged-router top-K should equal {C(i) : i ∈ TopK(h)}.
    # Sufficient condition: min logit over expected clusters > max over the rest.
    merged_lg = _merged_logits(logits, plan)              # [B, n_clusters]
    n_c = merged_lg.shape[1]

    expected_mask = torch.zeros(B, n_c, dtype=torch.bool, device=H.device)
    expected_mask.scatter_(1, topk_clusters, True)        # [B, n_clusters]

    # Min over expected, max over non-expected
    big = torch.full_like(merged_lg, float("inf"))
    small = torch.full_like(merged_lg, float("-inf"))
    min_expected = torch.where(expected_mask, merged_lg, big).min(dim=-1).values
    max_other = torch.where(~expected_mask, merged_lg, small).max(dim=-1).values
    # If there is no non-expected cluster (n_c == K), TS is vacuously true
    no_other = (~expected_mask).sum(dim=-1) == 0
    ts = (min_expected > max_other) | no_other

    # ---------- (EM) ----------
    # For each expected cluster C, sum p_k over members NOT in top-K.
    in_topk = torch.zeros(B, N, dtype=torch.bool, device=H.device)
    in_topk.scatter_(1, topk_idx, True)

    # For each (b, k): k is "expected cluster member but not top-K" iff
    #   label_of[k] ∈ expected_clusters(b)  AND  not in_topk[b, k]
    label_of_b = label_of.unsqueeze(0).expand(B, N)                # [B, N]
    is_in_expected_cluster = expected_mask.gather(1, label_of_b)   # [B, N]
    is_extra = is_in_expected_cluster & (~in_topk)                  # [B, N]
    xi = (probs * is_extra.float()).sum(dim=-1)                     # [B]
    em = xi <= delta_threshold

    return ConditionMasks(mc=mc, ts=ts, em=em, xi=xi)
