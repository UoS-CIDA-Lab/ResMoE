"""Stage 3 — Cluster construction and merge application.

Two clustering strategies:

  1. `complete_linkage_clusters` — classical bound-driven merging.
     Greedily merges the closest pair (max-linkage distance) until
     cutoff exceeded.

  2. `cert_aware_greedy_merge` — NEW (prototype contribution).
     At each step, considers all candidate merges below an ε bound
     threshold, simulates each, and picks the one that yields the
     HIGHEST certified-fraction on calibration data (Theorem 1
     conditions: MC ∧ TS ∧ EM). Bound tightness is used only as a
     gating constraint; the active objective is certifiable safety.

Both strategies hand off to `apply_merge` for the actual weight
averaging + log-sum-exp router merge.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass
class MergePlan:
    clusters: list[list[int]]   # partition of {0, ..., N-1}
    label_of: list[int]         # label_of[i] = cluster index containing i


def complete_linkage_clusters(
    delta_hat: torch.Tensor, epsilon: float
) -> MergePlan:
    """Agglomerative clustering with complete-linkage and cutoff ε.

    Two clusters can merge only if the MAX pairwise δ̂ between them is ≤ ε.
    This guarantees that for any cluster C and i ∈ C, the average expert
    E_C is within ε of E_i (Lemma 1 of framework doc).

    Args:
        delta_hat: [N, N] symmetric. ∞ for unverified or above-ε pairs,
                   verified bound for OK pairs. Diagonal ignored.
        epsilon:   merge threshold.

    Returns:
        MergePlan
    """
    N = delta_hat.shape[0]
    clusters: list[list[int]] = [[i] for i in range(N)]
    # Distance between clusters = max pairwise δ̂ within their union
    # (complete linkage). Start with the input matrix.
    cluster_dist = delta_hat.clone()
    cluster_dist.fill_diagonal_(math.inf)

    while True:
        # Find smallest cluster-distance ≤ ε
        flat_min = cluster_dist.flatten().min()
        if flat_min.item() > epsilon or len(clusters) == 1:
            break
        flat_idx = cluster_dist.flatten().argmin().item()
        a, b = divmod(flat_idx, cluster_dist.shape[0])
        if a == b:
            break
        # Merge cluster b into a
        if a > b:
            a, b = b, a
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
        # Rebuild cluster_dist
        n = len(clusters)
        new_dist = torch.full((n, n), math.inf)
        for i in range(n):
            for j in range(i + 1, n):
                # complete linkage: max over pairs
                pairs = [delta_hat[u, v].item()
                         for u in clusters[i] for v in clusters[j]]
                new_dist[i, j] = max(pairs)
                new_dist[j, i] = new_dist[i, j]
        cluster_dist = new_dist

    label_of = [0] * N
    for label, members in enumerate(clusters):
        for m in members:
            label_of[m] = label
    return MergePlan(clusters=clusters, label_of=label_of)


def _build_label_of(clusters: list[list[int]], N: int) -> list[int]:
    label_of = [0] * N
    for c_idx, members in enumerate(clusters):
        for m in members:
            label_of[m] = c_idx
    return label_of


def cert_aware_greedy_merge(
    moe,
    H: torch.Tensor,
    distance_matrix: torch.Tensor,
    target_n_clusters: int,
    max_bound: float,
    delta_em: float = 0.1,
    bound_weight: float = 0.0,
) -> "MergePlan":
    """Greedy merge that maximizes certified fraction at each step.

    At every step we score each candidate merge by:
        score(i,j) = cert_fraction(plan ∪ {merge(i,j)})
                     - bound_weight * normalized_bound(i,j)
    and pick the highest. Only candidates with max-linkage bound ≤
    `max_bound` are considered.

    bound_weight = 0  ⇒ pure cert-aware (no bound preference).
    bound_weight > 0  ⇒ trade off some cert% for tighter bound.

    Args:
        moe:               original MoE (needed for router_w, K)
        H:                 calibration hidden states [B, d]
        distance_matrix:   [N, N] pairwise bounds (any source)
        target_n_clusters: stop when reached
        max_bound:         hard cutoff — pairs with bound > this never
                           considered.
        delta_em:          EM threshold for Theorem 1 condition check
        bound_weight:      score mixing parameter
    """
    # Defer import to avoid circular dep at top
    from cert_moe.theorem1_conditions import check_conditions

    N = moe.cfg.n_experts
    K = moe.cfg.top_k
    W_g = moe.router.weight.detach()

    clusters: list[list[int]] = [[i] for i in range(N)]

    # Pre-normalize bounds for the mixed-objective score
    finite_bounds = distance_matrix[distance_matrix.isfinite()]
    bnd_scale = (
        finite_bounds.max().item() - finite_bounds.min().item() + 1e-8
    )

    with torch.no_grad():
        while len(clusters) > target_n_clusters:
            best = None  # (score, max_dist, a, b)
            n_c = len(clusters)
            for a in range(n_c):
                for b in range(a + 1, n_c):
                    # Max-linkage distance between the two clusters
                    ds = [
                        distance_matrix[u, v].item()
                        for u in clusters[a]
                        for v in clusters[b]
                    ]
                    max_dist = max(ds)
                    if max_dist > max_bound:
                        continue

                    # Tentative plan
                    trial = [
                        c for i, c in enumerate(clusters)
                        if i != a and i != b
                    ]
                    trial.append(clusters[a] + clusters[b])
                    trial_plan = MergePlan(
                        clusters=trial,
                        label_of=_build_label_of(trial, N),
                    )
                    masks = check_conditions(
                        H, W_g, trial_plan, K=K, delta_threshold=delta_em
                    )
                    cert_frac = masks.all_certified.float().mean().item()
                    bound_pen = max_dist / bnd_scale
                    score = cert_frac - bound_weight * bound_pen
                    cand = (score, max_dist, a, b)
                    if best is None or cand > best:
                        best = cand

            if best is None:
                break  # no more eligible merges

            _, _, a, b = best
            new_cluster = clusters[a] + clusters[b]
            clusters = [
                c for i, c in enumerate(clusters) if i != a and i != b
            ]
            clusters.append(new_cluster)

    return MergePlan(
        clusters=clusters,
        label_of=_build_label_of(clusters, N),
    )


def apply_merge(
    moe,
    plan: MergePlan,
    routing_freq: torch.Tensor,  # [N] usage frequency from calibration
):
    """Return a NEW ToyMoE-compatible structure with merged experts.

    - Each cluster C → one ExpertFFN whose weights are the routing-
      frequency-weighted average of its members.
    - Router rows merged via log-sum-exp: w_C = logsumexp({w_k : k ∈ C}).
    """
    # We import here to avoid circular dependency
    from cert_moe.toy_moe import ToyMoE, ToyMoEConfig, ExpertFFN

    n_new = len(plan.clusters)
    new_cfg = ToyMoEConfig(
        d_model=moe.cfg.d_model,
        d_ff=moe.cfg.d_ff,
        n_experts=n_new,
        top_k=min(moe.cfg.top_k, n_new),
        n_clone_pairs=0,
    )
    new_moe = ToyMoE.__new__(ToyMoE)
    nn.Module.__init__(new_moe)
    new_moe.cfg = new_cfg
    new_moe.router = nn.Linear(new_cfg.d_model, n_new, bias=False)
    new_moe.experts = nn.ModuleList(
        [ExpertFFN(new_cfg.d_model, new_cfg.d_ff) for _ in range(n_new)]
    )
    new_moe.clone_pairs = []

    with torch.no_grad():
        for c_idx, members in enumerate(plan.clusters):
            # Weights = normalized routing frequency
            freqs = routing_freq[members]
            if freqs.sum() == 0:
                w = torch.ones_like(freqs) / len(freqs)
            else:
                w = freqs / freqs.sum()

            # Weighted average of expert weights
            W1_avg = sum(w[i] * moe.experts[m].W1 for i, m in enumerate(members))
            W2_avg = sum(w[i] * moe.experts[m].W2 for i, m in enumerate(members))
            new_moe.experts[c_idx].W1.copy_(W1_avg)
            new_moe.experts[c_idx].W2.copy_(W2_avg)

            # Router: log-sum-exp merge
            cluster_logits_row = torch.logsumexp(
                moe.router.weight[members], dim=0
            )
            new_moe.router.weight[c_idx].copy_(cluster_logits_row)

    return new_moe
