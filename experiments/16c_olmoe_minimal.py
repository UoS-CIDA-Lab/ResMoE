"""Experiment 16c — Minimal-memory OLMoE Stage 3-4 from cache.

Goal: prove the framework runs end-to-end on OLMoE without OOM.

Memory tactics:
  - Single compression ratio (64 → 56, only 8 merges)
  - Subsample calibration to 32 tokens
  - Delete intermediate models after measurement
  - gc.collect between phases
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
    ToySwiGLUMoE, ToySwiGLUMoEConfig, SwiGLUExpert, apply_merge_swiglu,
)
from cert_moe.merge import (
    complete_linkage_clusters, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions


def build_moe_from_cache(cache):
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
    for i in range(cfg.n_experts):
        e = SwiGLUExpert(cfg.d_model, cfg.d_ff)
        with torch.no_grad():
            e.W1.copy_(cache["experts_W1"][i])
            e.W2.copy_(cache["experts_W2"][i])
            e.W3.copy_(cache["experts_W3"][i])
        moe.experts.append(e)
    moe.clone_pairs = []
    return moe


def cosine_swiglu(experts):
    import torch.nn.functional as F
    vecs = torch.stack([
        torch.cat([e.W1.detach().flatten(),
                   e.W2.detach().flatten(),
                   e.W3.detach().flatten()])
        for e in experts
    ])
    sim = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
    dist = 1.0 - sim
    dist.fill_diagonal_(0.0)
    return dist


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist()
                      if math.isfinite(d)))
    for eps in flat:
        plan = complete_linkage_clusters(dist, epsilon=eps)
        if len(plan.clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def main():
    cache_path = str(
        pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
    )
    print(f"Loading cache from {cache_path}...")
    cache = torch.load(cache_path, weights_only=True)

    print("Building MoE...")
    moe = build_moe_from_cache(cache)

    # Subsample H to 32 tokens to reduce memory
    H_full = cache["H"].clone()
    torch.manual_seed(0)
    if H_full.shape[0] > 32:
        idx = torch.randperm(H_full.shape[0])[:32]
        H = H_full[idx]
    else:
        H = H_full

    del cache, H_full
    gc.collect()
    print(f"  N={moe.cfg.n_experts}, K={moe.cfg.top_k}, "
          f"H={tuple(H.shape)}")

    # Routing freq
    with torch.no_grad():
        logits = H @ moe.router.weight.T
        _, topk_idx = logits.topk(moe.cfg.top_k, dim=-1)
    freq = torch.zeros(moe.cfg.n_experts)
    for k in range(moe.cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    print(f"  freq: min {freq.min():.0f}, max {freq.max():.0f}, "
          f"mean {freq.mean():.1f}")

    # Cosine distance
    print("\nBuilding cosine distance matrix...")
    d_cos = cosine_swiglu(moe.experts)

    target_n = 56   # 64 → 56, just 8 merges
    print(f"\n=== Compression 64 → {target_n} (8 merges) ===")

    # Original output baseline (compute once, keep small)
    with torch.no_grad():
        y_o, _ = moe(H)
    ref_norm = y_o.abs().mean().item()
    print(f"  Original output ||y||_2 mean: "
          f"{y_o.norm(dim=-1).mean().item():.2f}")

    # === Cosine plan ===
    print("\n--- Cosine merge ---")
    eps_c = find_cutoff(d_cos, target_n)
    plan_cos = complete_linkage_clusters(d_cos, epsilon=eps_c)
    merged_cos = apply_merge_swiglu(moe, plan_cos, freq)
    with torch.no_grad():
        y_cos, _ = merged_cos(H)
    rel_cos = (y_o - y_cos).abs().mean().item() / max(ref_norm, 1e-8)
    m_cos = check_conditions(H, moe.router.weight.detach(), plan_cos,
                             K=moe.cfg.top_k, delta_threshold=0.1)
    print(f"  rel_div {rel_cos*100:.1f}%, "
          f"MC {m_cos.mc.float().mean()*100:.1f}%, "
          f"TS {m_cos.ts.float().mean()*100:.1f}%, "
          f"EM {m_cos.em.float().mean()*100:.1f}%, "
          f"cert {m_cos.all_certified.float().mean()*100:.1f}%")

    # Save cosine results
    cos_results = {
        "rel_div": rel_cos,
        "cert": m_cos.all_certified.float().mean().item(),
        "ts": m_cos.ts.float().mean().item(),
        "em": m_cos.em.float().mean().item(),
    }
    del merged_cos, y_cos, m_cos
    gc.collect()

    # === Cert-aware plan ===
    print("\n--- Cert-aware merge (greedy, 8 steps over ~2000 candidates)"
          " ---")
    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001
    plan_ca = cert_aware_greedy_merge(
        moe, H, d_cos,
        target_n_clusters=target_n,
        max_bound=max_b, delta_em=0.1, bound_weight=0.0,
    )
    merged_ca = apply_merge_swiglu(moe, plan_ca, freq)
    with torch.no_grad():
        y_ca, _ = merged_ca(H)
    rel_ca = (y_o - y_ca).abs().mean().item() / max(ref_norm, 1e-8)
    m_ca = check_conditions(H, moe.router.weight.detach(), plan_ca,
                            K=moe.cfg.top_k, delta_threshold=0.1)
    print(f"  rel_div {rel_ca*100:.1f}%, "
          f"MC {m_ca.mc.float().mean()*100:.1f}%, "
          f"TS {m_ca.ts.float().mean()*100:.1f}%, "
          f"EM {m_ca.em.float().mean()*100:.1f}%, "
          f"cert {m_ca.all_certified.float().mean()*100:.1f}%")

    # Summary comparison
    print("\n=== Summary 64 → 56 ===")
    print(f"  cosine    : rel_div {cos_results['rel_div']*100:5.1f}%, "
          f"cert {cos_results['cert']*100:5.1f}%, "
          f"TS {cos_results['ts']*100:5.1f}%, "
          f"EM {cos_results['em']*100:5.1f}%")
    print(f"  cert-aware: rel_div {rel_ca*100:5.1f}%, "
          f"cert {m_ca.all_certified.float().mean()*100:5.1f}%, "
          f"TS {m_ca.ts.float().mean()*100:5.1f}%, "
          f"EM {m_ca.em.float().mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
