"""Experiment 17 — Baseline comparison on cached OLMoE layer 0.

Runs every merging method through the SAME Stage-3 pipeline
(distance matrix → clustering → weight averaging + log-sum-exp router
merge) so the only thing that varies is HOW expert pairs are ranked.

Methods compared:
  cosine     — weight-space cosine distance (B3)
  mc_smoe    — routing co-activation + weight cosine (B4, Li+24 spirit)
  hc_smoe    — empirical output-space L2 distance (B5, Chen+24 spirit)
  freq_prune — drop least-used experts, no merging (B6)
  cert-aware — greedy merge maximizing Theorem-1 certified fraction (ours)

For each method and compression ratio we report rel_div (output
divergence vs original) and the Theorem-1 condition pass rates
MC / TS / EM / cert.

Memory-lean: loads the 1.6GB cache tensor-by-tensor, builds the MoE
with empty (not randn) parameters, and subsamples calibration tokens.
"""
from __future__ import annotations

import sys
import pathlib
import math
import gc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from cert_moe.toy_swiglu_moe import (
    ToySwiGLUMoE, ToySwiGLUMoEConfig, apply_merge_swiglu,
)
from cert_moe.merge import (
    complete_linkage_clusters, cert_aware_greedy_merge,
)
from cert_moe.baselines import (
    cosine_distance_matrix, mc_smoe_distance_matrix,
    hc_smoe_distance_matrix, frequency_pruning_plan,
)
from cert_moe.theorem1_conditions import check_conditions


class LightSwiGLUExpert(nn.Module):
    """SwiGLU expert with empty (not randn) weight allocation."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W3 = nn.Parameter(torch.empty(d_model, d_ff))

    def forward(self, h):
        from cert_moe.swiglu_bounds import silu
        return (silu(h @ self.W1.T) * (h @ self.W2.T)) @ self.W3.T


def build_lean_moe(cache_path):
    cache = torch.load(cache_path, weights_only=True, map_location="cpu")
    cfg = ToySwiGLUMoEConfig(
        d_model=cache["d_model"], d_ff=cache["d_ff"],
        n_experts=cache["n_experts"], top_k=cache["top_k"],
        n_clone_pairs=0,
    )
    moe = ToySwiGLUMoE.__new__(ToySwiGLUMoE)
    nn.Module.__init__(moe)
    moe.cfg = cfg
    moe.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
    with torch.no_grad():
        moe.router.weight.copy_(cache["router_weight"])
    moe.experts = nn.ModuleList()
    W1s, W2s, W3s = cache["experts_W1"], cache["experts_W2"], cache["experts_W3"]
    H = cache["H"].clone()
    del cache
    gc.collect()
    for i in range(cfg.n_experts):
        e = LightSwiGLUExpert(cfg.d_model, cfg.d_ff)
        with torch.no_grad():
            e.W1.copy_(W1s[i])
            e.W2.copy_(W2s[i])
            e.W3.copy_(W3s[i])
        moe.experts.append(e)
    moe.clone_pairs = []
    del W1s, W2s, W3s
    gc.collect()
    return moe, H


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist()
                      if math.isfinite(d)))
    for eps in flat:
        plan = complete_linkage_clusters(dist, epsilon=eps)
        if len(plan.clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def evaluate(moe, H, freq, plan, ref):
    """Return dict of rel_div + Theorem-1 condition pass rates."""
    merged = apply_merge_swiglu(moe, plan, freq)
    with torch.no_grad():
        y_m, _ = merged(H)
        y_o, _ = moe(H)
    rel = (y_o - y_m).abs().mean().item() / max(ref, 1e-8)
    m = check_conditions(H, moe.router.weight.detach(), plan,
                         K=moe.cfg.top_k, delta_threshold=0.1)
    del merged, y_m
    gc.collect()
    return {
        "n": len(plan.clusters),
        "rel": rel,
        "mc": m.mc.float().mean().item(),
        "ts": m.ts.float().mean().item(),
        "em": m.em.float().mean().item(),
        "cert": m.all_certified.float().mean().item(),
    }


def main():
    cache_path = str(
        pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
    )
    print("Building lean MoE from cache...")
    moe, H = build_lean_moe(cache_path)
    K = moe.cfg.top_k
    print(f"  N={moe.cfg.n_experts}, K={K}, d={moe.cfg.d_model}")

    torch.manual_seed(0)
    if H.shape[0] > 32:
        H = H[torch.randperm(H.shape[0])[:32]]
    print(f"  calibration H={tuple(H.shape)}")

    with torch.no_grad():
        logits = H @ moe.router.weight.T
        _, topk_idx = logits.topk(K, dim=-1)
    freq = torch.zeros(moe.cfg.n_experts)
    for k in range(K):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    print(f"  expert usage: min {freq.min():.0f}, max {freq.max():.0f}, "
          f"mean {freq.mean():.1f}")

    with torch.no_grad():
        y_o, _ = moe(H)
    ref = y_o.abs().mean().item()
    print(f"  baseline ||y|| mean {y_o.norm(dim=-1).mean().item():.3f}")
    del y_o
    gc.collect()

    # Pre-compute distance matrices for the distance-based baselines.
    print("\nComputing distance matrices...")
    dmats = {
        "cosine": cosine_distance_matrix(moe.experts),
        "mc_smoe": mc_smoe_distance_matrix(
            moe.experts, H, moe.router.weight.detach(), K),
        "hc_smoe": hc_smoe_distance_matrix(moe.experts, H),
    }
    for name, d in dmats.items():
        finite = d[d.isfinite() & (d > 0)]
        print(f"  {name:8s}: range [{finite.min():.4f}, {finite.max():.4f}]")

    for target_n in (48, 32):
        print(f"\n{'='*72}")
        print(f"  Compression  64 → {target_n}")
        print(f"{'='*72}")
        rows = []

        # Distance-based heuristics via complete-linkage
        for name, d in dmats.items():
            eps = find_cutoff(d, target_n)
            plan = complete_linkage_clusters(d, epsilon=eps)
            rows.append((name, evaluate(moe, H, freq, plan, ref)))

        # Frequency pruning (B6) — its own plan builder
        plan_fp = frequency_pruning_plan(
            moe.experts, H, moe.router.weight.detach(), K, target_n)
        rows.append(("freq_prune", evaluate(moe, H, freq, plan_fp, ref)))

        # Cert-aware (ours): cosine as gating distance, no bound preference
        d_gate = dmats["cosine"]
        max_b = d_gate[d_gate.isfinite()].max().item() * 1.001
        plan_ca = cert_aware_greedy_merge(
            moe, H, d_gate, target_n_clusters=target_n,
            max_bound=max_b, delta_em=0.1, bound_weight=0.0,
        )
        rows.append(("cert-aware", evaluate(moe, H, freq, plan_ca, ref)))

        # Report
        print(f"\n  {'method':12s} {'n':>4s} {'rel_div':>8s} "
              f"{'MC':>6s} {'TS':>6s} {'EM':>6s} {'cert':>6s}")
        print("  " + "-" * 54)
        for name, r in rows:
            print(f"  {name:12s} {r['n']:>4d} {r['rel']*100:>7.1f}% "
                  f"{r['mc']*100:>5.1f}% {r['ts']*100:>5.1f}% "
                  f"{r['em']*100:>5.1f}% {r['cert']*100:>5.1f}%")


if __name__ == "__main__":
    main()
