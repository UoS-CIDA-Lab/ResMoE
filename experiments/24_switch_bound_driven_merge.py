"""Experiment 24 — Is bound-driven merging really "random"? (CROWN vs α-CROWN)

RESEARCH_OVERVIEW §5.2 claims the verified bound's pair RANKING is
uncorrelated with empirical divergence on Switch, so "bound-driven merging
is nearly random." But exp 23 found α-CROWN full-box has Spearman +0.585
with empirical — not random. Hypothesis: the "random" claim was measured
with plain CROWN; α-CROWN restores a usable merge signal.

This tests it directly. For Switch-base-8 layer 1 we build five pairwise
distance matrices and run the SAME complete-linkage merge from each:
  - crown      : plain CROWN bound (full box)
  - alpha_crown: α-CROWN bound (full box)
  - cosine     : weight cosine (empirical baseline)
  - empirical  : true pairwise output divergence (ORACLE upper bound on
                 how good this clustering family can do)
  - random     : average over many random partitions (the "random" floor)

Metric: output rel_div at the MoE layer after merge (lower = better merge).
Plus Spearman(bound, empirical) for crown and alpha_crown to pinpoint where
the discrimination comes from.

Needs HF_HUB_DISABLE_XET=1 (Switch already downloaded).
"""
from __future__ import annotations

import sys
import pathlib
import math
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import (
    crown_diff_bound, alpha_crown_diff_bound, empirical_pair_diff,
)
from cert_moe.baselines import cosine_distance_matrix
from cert_moe.merge import (
    complete_linkage_clusters, apply_merge, MergePlan, _build_label_of,
)
from cert_moe.theorem1_conditions import check_conditions

PATH = "encoder.block.1.layer.1.mlp"
CALIB = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "Machine learning models predict outputs from inputs.",
    "Neural networks consist of layers of neurons.",
    "Transformers use attention mechanisms.",
    "Mixture of experts increases model capacity.",
    "Quantum computers may revolutionize computation.",
    "Climate change is a pressing global concern.",
    "Mathematics describes the universe.",
    "Music transcends language barriers.",
    "Education shapes future generations.",
    "Technology drives social change.",
    "Sports build character and camaraderie.",
    "Art expresses what words cannot.",
    "History informs our present and future.",
] * 6


def spearman(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra -= ra.mean(); rb -= rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm()).clamp(min=1e-12)).item()


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def rel_div(moe, plan, H, freq):
    merged = apply_merge(moe, plan, freq)
    with torch.no_grad():
        y_o, _ = moe(H)
        y_m, _ = merged(H)
    return (y_o - y_m).abs().mean().item() / y_o.abs().mean().item()


def random_plan(n_experts, target_n, gen):
    labels = torch.randint(0, target_n, (n_experts,), generator=gen).tolist()
    # ensure all target_n clusters non-empty by forcing first target_n experts
    for c in range(target_n):
        labels[c] = c
    clusters = [[i for i in range(n_experts) if labels[i] == c]
                for c in range(target_n)]
    clusters = [c for c in clusters if c]
    return MergePlan(clusters=clusters,
                     label_of=_build_label_of(clusters, n_experts))


def main():
    print("Loading Switch-base-8 layer 1...")
    moe, model, tok = load_switch_block()
    N, K = moe.cfg.n_experts, moe.cfg.top_k
    dtype = moe.router.weight.dtype

    H = collect_hidden_states(model, tok, PATH, CALIB, batch_size=8).to(dtype)
    print(f"Calibration: {H.shape[0]} tokens")
    with torch.no_grad():
        logits = H @ moe.router.weight.T
    _, topk_idx = logits.topk(K, dim=-1)
    freq = torch.zeros(N)
    for k in range(K):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))

    h_lo, h_hi = H.min(0).values - 0.05, H.max(0).values + 0.05

    # Build distance matrices
    print("Building distance matrices (crown, alpha_crown, cosine, empirical)...")
    INF = math.inf
    d_crown = torch.full((N, N), INF); d_crown.fill_diagonal_(0)
    d_acrown = torch.full((N, N), INF); d_acrown.fill_diagonal_(0)
    d_emp = torch.full((N, N), INF); d_emp.fill_diagonal_(0)
    pair_list = []
    for i, j in combinations(range(N), 2):
        ei, ej = moe.experts[i], moe.experts[j]
        bc = crown_diff_bound(ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi)
        ba = alpha_crown_diff_bound(ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi,
                                    n_iters=20)
        be = empirical_pair_diff(ei, ej, H)
        d_crown[i, j] = d_crown[j, i] = bc
        d_acrown[i, j] = d_acrown[j, i] = ba
        d_emp[i, j] = d_emp[j, i] = be
        pair_list.append((bc, ba, be))
    d_cos = cosine_distance_matrix(moe.experts)

    bc_l = [p[0] for p in pair_list]
    ba_l = [p[1] for p in pair_list]
    be_l = [p[2] for p in pair_list]
    print(f"\n  Spearman(crown,       empirical) = {spearman(bc_l, be_l):+.3f}")
    print(f"  Spearman(alpha_crown, empirical) = {spearman(ba_l, be_l):+.3f}")
    print(f"  Spearman(cosine,      empirical) = "
          f"{spearman([d_cos[i, j].item() for i, j in combinations(range(N), 2)], be_l):+.3f}")

    methods = {
        "crown": d_crown, "alpha_crown": d_acrown,
        "cosine": d_cos, "empirical(oracle)": d_emp,
    }

    gen = torch.Generator().manual_seed(0)
    for target_n in (6, 4):
        print(f"\n{'='*60}")
        print(f"  Switch 8 → {target_n}: merge quality (rel_div, lower=better)")
        print(f"{'='*60}")
        for name, dist in methods.items():
            eps = find_cutoff(dist, target_n)
            plan = complete_linkage_clusters(dist, epsilon=eps)
            rd = rel_div(moe, plan, H, freq)
            cert = check_conditions(H, moe.router.weight.detach(), plan,
                                    K=K, delta_threshold=0.1
                                    ).all_certified.float().mean().item()
            print(f"  {name:18s} rel_div {rd*100:6.2f}%   cert {cert*100:5.1f}%")
        # random floor
        rds = [rel_div(moe, random_plan(N, target_n, gen), H, freq)
               for _ in range(20)]
        rds.sort()
        print(f"  {'random (20 plans)':18s} rel_div {sum(rds)/len(rds)*100:6.2f}% "
              f"(best {rds[0]*100:.2f}%, worst {rds[-1]*100:.2f}%)")

    print(f"\n{'='*60}")
    print("  VERDICT")
    print("  - If alpha_crown rel_div ≈ cosine/empirical and ≪ random,")
    print("    bound-driven merging is NOT random (doc claim was CROWN-only).")
    print("  - Compare crown vs alpha_crown to locate the difference.")


if __name__ == "__main__":
    main()
