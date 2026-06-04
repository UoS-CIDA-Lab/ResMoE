"""Experiment 09 — Stage 3 cert-aware merging on Switch-base-8.

The crucial test: does cert-aware merging dominate on a real model?
Toy showed cert-aware wins at high compression (16→8). Here we test
8→6 and 8→4 on a pretrained MoE layer.

Comparisons (all use the same complete-linkage / cert-aware machinery,
just different distance signals):
  - cosine         : weight-space cosine similarity
  - hc_smoe        : empirical output-space distance
  - bound (CROWN)  : verified equivalence bound
  - cert-aware     : greedy maximize cert%

Metrics:
  - Cert% at each compression
  - Output divergence at the MoE layer (vs original)
  - Theorem-1 per-condition pass rates
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from itertools import combinations
import math

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import pair_diff_bound
from cert_moe.merge import (
    complete_linkage_clusters, apply_merge, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions
from cert_moe.baselines import cosine_distance_matrix, hc_smoe_distance_matrix


SAMPLES = [
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


def build_bound_matrix(moe, H, h_lo, h_hi, fast=True):
    """Pairwise CROWN bound matrix (no α-CROWN optimization for speed)."""
    N = moe.cfg.n_experts
    method = "crown" if fast else "alpha_crown"
    delta = torch.full((N, N), math.inf)
    delta.fill_diagonal_(0.0)
    for i, j in combinations(range(N), 2):
        b = pair_diff_bound(
            moe.experts[i], moe.experts[j], h_lo, h_hi,
            method=method, alpha_iters=10,
        )
        delta[i, j] = b
        delta[j, i] = b
    return delta


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        plan = complete_linkage_clusters(dist, epsilon=eps)
        if len(plan.clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def measure_at_moe_layer(moe, plan, H, freq):
    """Output divergence and cert% at the MoE layer."""
    merged = apply_merge(moe, plan, freq)
    with torch.no_grad():
        y_orig, _ = moe(H)
        y_merged, _ = merged(H)
    diff = (y_orig - y_merged).abs()
    masks = check_conditions(
        H, moe.router.weight.detach(), plan, K=moe.cfg.top_k,
        delta_threshold=0.1,
    )
    return {
        "merged": merged,
        "div_mean": diff.mean().item(),
        "div_max_per_token": diff.max(-1).values.mean().item(),
        "rel_div": diff.mean().item() / y_orig.abs().mean().item(),
        "mc": masks.mc.float().mean().item(),
        "ts": masks.ts.float().mean().item(),
        "em": masks.em.float().mean().item(),
        "cert": masks.all_certified.float().mean().item(),
    }


def main():
    print("Loading Switch-base-8 first MoE block...")
    moe, model, tok = load_switch_block()

    print("\nCollecting calibration hidden states...")
    H = collect_hidden_states(
        model, tok, "encoder.block.1.layer.1.mlp",
        SAMPLES, batch_size=8,
    ).to(moe.router.weight.dtype)
    if H.shape[0] > 1024:
        H = H[torch.randperm(H.shape[0])[:1024]]
    print(f"Calibration: {H.shape}")

    # Frequency for merge weights
    with torch.no_grad():
        logits = H @ moe.router.weight.T
    _, topk_idx = logits.topk(moe.cfg.top_k, dim=-1)
    freq = torch.zeros(moe.cfg.n_experts)
    for k in range(moe.cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    print(f"Expert usage: {freq.long().tolist()}")

    # Distance matrices
    print("\nBuilding distance matrices...")
    h_lo = H.min(0).values - 0.05
    h_hi = H.max(0).values + 0.05
    print("  cosine  ...", end=" ", flush=True)
    d_cosine = cosine_distance_matrix(moe.experts)
    print("done")
    print("  hc_smoe ...", end=" ", flush=True)
    d_hc = hc_smoe_distance_matrix(moe.experts, H)
    print("done")
    print("  bound (CROWN, no α) ...", end=" ", flush=True)
    d_bound = build_bound_matrix(moe, H, h_lo, h_hi, fast=True)
    print("done")
    methods = {"cosine": d_cosine, "hc_smoe": d_hc, "bound": d_bound}

    # Compression sweep
    target_ns = [6, 4]
    print(f"\n{'='*88}")
    print("Compression sweep on Switch-base-8")
    print(f"{'='*88}")

    for target_n in target_ns:
        print(f"\n  ── 8 → {target_n} ─" + "─" * 70)
        header = (f"{'method':25s} | {'div_mean':>9s} | {'rel_div':>8s}"
                  f" | {'MC%':>5s} | {'TS%':>5s} | {'EM%':>5s} | {'cert%':>6s}")
        print(header)
        print("-" * len(header))

        # Complete-linkage for each baseline
        for name, dist in methods.items():
            eps = find_cutoff(dist, target_n)
            plan = complete_linkage_clusters(dist, epsilon=eps)
            r = measure_at_moe_layer(moe, plan, H, freq)
            print(f"  {name:25s} | {r['div_mean']:>9.3f} | "
                  f"{r['rel_div']*100:>7.2f}% | "
                  f"{r['mc']*100:>4.1f}% | {r['ts']*100:>4.1f}% | "
                  f"{r['em']*100:>4.1f}% | {r['cert']*100:>5.1f}%")

        # Cert-aware (greedy by cert%) for each distance source
        for src_name, dist in methods.items():
            max_b = dist[dist.isfinite()].max().item() * 1.001
            plan = cert_aware_greedy_merge(
                moe, H, dist,
                target_n_clusters=target_n,
                max_bound=max_b,
                delta_em=0.1,
                bound_weight=0.0,
            )
            if len(plan.clusters) > target_n:
                print(f"  cert-aware({src_name}): could not reach {target_n}")
                continue
            r = measure_at_moe_layer(moe, plan, H, freq)
            label = f"cert-aware({src_name})"
            print(f"  {label:25s} | {r['div_mean']:>9.3f} | "
                  f"{r['rel_div']*100:>7.2f}% | "
                  f"{r['mc']*100:>4.1f}% | {r['ts']*100:>4.1f}% | "
                  f"{r['em']*100:>4.1f}% | {r['cert']*100:>5.1f}%")

    print("\n" + "=" * 88)
    print("Key questions:")
    print("  - Does cert-aware retain cert% > 0 when baselines hit 0?")
    print("  - Does cert-aware win on relative divergence at 8→4?")
    print("  - Which baselines preserve clone-style structure on a real model?")
    print("=" * 88)


if __name__ == "__main__":
    main()
