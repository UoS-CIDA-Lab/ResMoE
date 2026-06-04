"""Heuristic baselines for MoE expert merging — for direct comparison
with the certified framework.

Each baseline produces a distance matrix δ ∈ R^{N×N} that we feed to
the same `complete_linkage_clusters` + `apply_merge` pipeline as our
certified method, so the only thing that changes is HOW pairs are
ranked.

Implemented baselines:
  B3  Cosine — weight-space cosine distance (simplest)
  B4  MC-SMoE-style — routing-co-activation distance (Li et al., 2024)
  B5  HC-SMoE-style — activation-space distance via expert outputs
                       (Chen et al., 2024)
  B6  Frequency pruning — drop least-used experts, no merging

These are simplified re-implementations matching the spirit of each
method; faithful re-impl would track each paper's specifics.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  B3 — Cosine similarity                                                      #
# --------------------------------------------------------------------------- #

def cosine_distance_matrix(experts) -> torch.Tensor:
    """Distance = 1 - cosine_sim of concatenated (W1, W2) flat vectors."""
    N = len(experts)
    vecs = torch.stack([
        torch.cat([e.W1.detach().flatten(), e.W2.detach().flatten()])
        for e in experts
    ])
    vecs = F.normalize(vecs, dim=-1)
    sim = vecs @ vecs.T              # [N, N] — avoids [N, N, P] broadcast OOM
    dist = 1.0 - sim
    dist.fill_diagonal_(0.0)
    return dist


# --------------------------------------------------------------------------- #
#  B4 — MC-SMoE-style (routing co-activation)                                  #
# --------------------------------------------------------------------------- #

def mc_smoe_distance_matrix(
    experts, H: torch.Tensor, router_w: torch.Tensor, K: int,
    weight_mix: float = 0.5,
) -> torch.Tensor:
    """Distance combines two terms (simplified MC-SMoE):
      (1) Routing co-activation: pairs that route similar tokens are
          considered similar.  d_route(i,j) = 1 - Jaccard(top-K membership)
      (2) Weight cosine on (W1, W2): backup signal.
    Final: weight_mix * d_route + (1 - weight_mix) * d_weight.
    """
    N = len(experts)
    logits = H @ router_w.T
    _, topk_idx = logits.topk(K, dim=-1)               # [B, K]
    in_topk = torch.zeros(H.shape[0], N, dtype=torch.bool)
    in_topk.scatter_(1, topk_idx, True)                # [B, N]

    # Jaccard on token-sets per expert
    inter = (in_topk.float().T @ in_topk.float())      # [N, N]
    sums = in_topk.float().sum(0)                       # [N]
    union = sums.unsqueeze(0) + sums.unsqueeze(1) - inter
    jaccard = inter / union.clamp(min=1.0)
    d_route = 1.0 - jaccard

    # Co-activation: experts that NEVER share tokens get d_route = 1
    # (Jaccard = 0). MC-SMoE specifically merges co-active experts.
    # We flip to "we prefer NON-co-active pairs to merge" - actually
    # MC-SMoE merges experts that handle SIMILAR token populations
    # (high Jaccard), so d_route = 1 - Jaccard is correct: small dist
    # = similar routing pattern = merge candidate.

    # Weight cosine
    d_weight = cosine_distance_matrix(experts)

    dist = weight_mix * d_route + (1.0 - weight_mix) * d_weight
    dist.fill_diagonal_(0.0)
    return dist


# --------------------------------------------------------------------------- #
#  B5 — HC-SMoE-style (activation-space distance)                              #
# --------------------------------------------------------------------------- #

def hc_smoe_distance_matrix(
    experts, H: torch.Tensor, sample_cap: int = 256,
) -> torch.Tensor:
    """Distance = mean ||E_i(h) - E_j(h)||_2 over calibration samples.

    This is empirical pairwise output divergence — what HC-SMoE roughly
    uses (activation-space hierarchical clustering).
    """
    N = len(experts)
    if H.shape[0] > sample_cap:
        idx = torch.randperm(H.shape[0])[:sample_cap]
        H = H[idx]
    with torch.no_grad():
        outs = torch.stack([e(H) for e in experts])    # [N, B, d]
    # Pairwise L2 distance averaged over batch
    diff = outs.unsqueeze(0) - outs.unsqueeze(1)        # [N, N, B, d]
    dist = diff.norm(dim=-1).mean(dim=-1)                # [N, N]
    dist.fill_diagonal_(0.0)
    return dist


# --------------------------------------------------------------------------- #
#  B6 — Frequency pruning                                                      #
# --------------------------------------------------------------------------- #

def frequency_pruning_plan(
    experts, H: torch.Tensor, router_w: torch.Tensor, K: int,
    target_n_experts: int,
):
    """Keep the top `target_n_experts` by routing frequency; prune rest.

    Pruned experts have their router rows set to -inf so they're never
    selected. We model this as a merge plan that collapses all pruned
    experts into one "graveyard" cluster (but never routed to in
    practice).
    """
    from cert_moe.merge import MergePlan
    N = len(experts)
    logits = H @ router_w.T
    _, topk_idx = logits.topk(K, dim=-1)
    freq = torch.zeros(N)
    for k in range(K):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    keep = freq.argsort(descending=True)[:target_n_experts].tolist()
    drop = [i for i in range(N) if i not in keep]
    # Layout: kept experts as singletons + all dropped in one cluster
    clusters = [[i] for i in keep] + [drop]
    if not drop:
        clusters = [[i] for i in keep]
    label_of = [0] * N
    for c_idx, members in enumerate(clusters):
        for m in members:
            label_of[m] = c_idx
    return MergePlan(clusters=clusters, label_of=label_of)


# --------------------------------------------------------------------------- #
#  Unified entry point                                                         #
# --------------------------------------------------------------------------- #

def baseline_distance_matrix(
    name: str, experts, H: torch.Tensor,
    router_w: torch.Tensor | None = None, K: int = 2,
) -> torch.Tensor:
    """Dispatch to the chosen baseline distance matrix."""
    if name == "cosine":
        return cosine_distance_matrix(experts)
    if name == "mc_smoe":
        assert router_w is not None
        return mc_smoe_distance_matrix(experts, H, router_w, K)
    if name == "hc_smoe":
        return hc_smoe_distance_matrix(experts, H)
    raise ValueError(f"Unknown baseline: {name}")
