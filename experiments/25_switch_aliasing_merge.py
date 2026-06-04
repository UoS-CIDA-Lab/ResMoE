"""Experiment 25 — Aliasing merge: does output-equivalence become the RIGHT
criterion when we drop-and-route instead of weight-average?

Experiment 24 bombshell: clustering by true output divergence (empirical
"oracle") gave the WORST weight-averaging merges, while cosine (weight
geometry) gave the best. Reason: weight-averaging quality depends on weight
geometry, not output equivalence — two functionally-equivalent experts with
different weight parameterizations average to garbage.

The fix (and the only place verification's worst-case guarantee could
matter): ALIASING merge. For each cluster, keep ONE representative expert's
weights verbatim and route the whole cluster to it. Then the merge error for
a dropped member j is exactly ||E_rep(h) - E_j(h)|| — the pairwise output
divergence that empirical/CROWN measure. Prediction: under aliasing the
empirical oracle should flip from WORST to BEST, and the α-CROWN bound (which
ranks pairs by output equivalence) should pick good merges too.

We apply the SAME plans from exp 24 two ways — averaging vs aliasing — and
compare rel_div.

Needs HF_HUB_DISABLE_XET=1 (Switch already downloaded).
"""
from __future__ import annotations

import sys
import pathlib
import math
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import alpha_crown_diff_bound, empirical_pair_diff
from cert_moe.baselines import cosine_distance_matrix
from cert_moe.merge import complete_linkage_clusters, apply_merge
from cert_moe.toy_moe import ToyMoE, ToyMoEConfig, ExpertFFN

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


def apply_merge_alias(moe, plan, freq):
    """Aliasing merge: each cluster keeps its highest-frequency member's
    weights verbatim (no averaging); router merged via log-sum-exp."""
    n_new = len(plan.clusters)
    cfg = ToyMoEConfig(d_model=moe.cfg.d_model, d_ff=moe.cfg.d_ff,
                       n_experts=n_new, top_k=min(moe.cfg.top_k, n_new),
                       n_clone_pairs=0)
    new = ToyMoE.__new__(ToyMoE)
    nn.Module.__init__(new)
    new.cfg = cfg
    new.router = nn.Linear(cfg.d_model, n_new, bias=False)
    new.experts = nn.ModuleList(
        [ExpertFFN(cfg.d_model, cfg.d_ff) for _ in range(n_new)])
    new.clone_pairs = []
    with torch.no_grad():
        for c, members in enumerate(plan.clusters):
            fr = torch.tensor([freq[m].item() for m in members])
            rep = members[int(fr.argmax())]
            new.experts[c].W1.copy_(moe.experts[rep].W1)
            new.experts[c].W2.copy_(moe.experts[rep].W2)
            new.router.weight[c].copy_(
                torch.logsumexp(moe.router.weight[members], dim=0))
    return new


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def rel_div(moe, merged, H):
    with torch.no_grad():
        y_o, _ = moe(H)
        y_m, _ = merged(H)
    return (y_o - y_m).abs().mean().item() / y_o.abs().mean().item()


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

    print("Building distance matrices...")
    INF = math.inf
    d_acrown = torch.full((N, N), INF); d_acrown.fill_diagonal_(0)
    d_emp = torch.full((N, N), INF); d_emp.fill_diagonal_(0)
    for i, j in combinations(range(N), 2):
        ei, ej = moe.experts[i], moe.experts[j]
        d_acrown[i, j] = d_acrown[j, i] = alpha_crown_diff_bound(
            ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi, n_iters=20)
        d_emp[i, j] = d_emp[j, i] = empirical_pair_diff(ei, ej, H)
    d_cos = cosine_distance_matrix(moe.experts)

    methods = {"cosine": d_cos, "alpha_crown": d_acrown,
               "empirical(oracle)": d_emp}

    for target_n in (6, 4):
        print(f"\n{'='*64}")
        print(f"  Switch 8 → {target_n}: rel_div  (averaging vs ALIASING)")
        print(f"{'='*64}")
        print(f"  {'method':18s} {'averaging':>11s} {'aliasing':>11s}")
        print("  " + "-" * 42)
        for name, dist in methods.items():
            eps = find_cutoff(dist, target_n)
            plan = complete_linkage_clusters(dist, epsilon=eps)
            rd_avg = rel_div(moe, apply_merge(moe, plan, freq), H)
            rd_ali = rel_div(moe, apply_merge_alias(moe, plan, freq), H)
            print(f"  {name:18s} {rd_avg*100:>10.2f}% {rd_ali*100:>10.2f}%")

    print(f"\n{'='*64}")
    print("  PREDICTION: under aliasing, empirical(oracle) should be BEST")
    print("  (its error = pairwise output divergence, exactly what it ranks).")
    print("  If so → output-equivalence is the right criterion for aliasing,")
    print("  and verification's worst-case guarantee finally has a purpose.")


if __name__ == "__main__":
    main()
