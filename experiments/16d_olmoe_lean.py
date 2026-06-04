"""Experiment 16d — Memory-lean OLMoE Stage 3-4.

Aggressive memory tactics:
  - Load cache tensors one at a time using torch.load(map_location='cpu')
  - Build MoE by directly copying into pre-allocated parameters
  - Free cache file handle ASAP
  - Use torch.empty (not randn) for parameter allocation
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
    W1s = cache["experts_W1"]
    W2s = cache["experts_W2"]
    W3s = cache["experts_W3"]
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


def cosine_swiglu(experts):
    import torch.nn.functional as F
    vecs = torch.stack([
        torch.cat([e.W1.detach().flatten(),
                   e.W2.detach().flatten(),
                   e.W3.detach().flatten()])
        for e in experts
    ])
    vecs = F.normalize(vecs, dim=-1)
    sim = vecs @ vecs.T            # [N, N] — avoids [N, N, P] broadcast OOM
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
    print("Building lean MoE from cache...")
    moe, H = build_lean_moe(cache_path)
    print(f"  Built. N={moe.cfg.n_experts}, K={moe.cfg.top_k}")

    # Subsample H to 32 tokens
    torch.manual_seed(0)
    if H.shape[0] > 32:
        idx = torch.randperm(H.shape[0])[:32]
        H = H[idx]
    print(f"  H={tuple(H.shape)}")

    # Routing freq
    with torch.no_grad():
        logits = H @ moe.router.weight.T
        _, topk_idx = logits.topk(moe.cfg.top_k, dim=-1)
    freq = torch.zeros(moe.cfg.n_experts)
    for k in range(moe.cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    print(f"  freq: min {freq.min():.0f}, max {freq.max():.0f}")

    # Cosine
    print("\nCosine distance matrix...")
    d_cos = cosine_swiglu(moe.experts)
    print(f"  d_cos[non-diag] range: "
          f"[{d_cos[d_cos > 0].min():.4f}, {d_cos.max():.4f}]")

    target_n = 56
    print(f"\n=== 64 → {target_n} (8 merges) ===")

    # Original output baseline
    with torch.no_grad():
        y_o, _ = moe(H)
    ref = y_o.abs().mean().item()
    print(f"  baseline ||y|| mean {y_o.norm(dim=-1).mean().item():.2f}")

    # Cosine
    eps_c = find_cutoff(d_cos, target_n)
    plan_cos = complete_linkage_clusters(d_cos, epsilon=eps_c)
    merged_cos = apply_merge_swiglu(moe, plan_cos, freq)
    with torch.no_grad():
        y_c, _ = merged_cos(H)
    rel_c = (y_o - y_c).abs().mean().item() / max(ref, 1e-8)
    m_c = check_conditions(H, moe.router.weight.detach(), plan_cos,
                           K=moe.cfg.top_k, delta_threshold=0.1)
    print(f"\ncosine: rel_div {rel_c*100:.1f}%, "
          f"MC {m_c.mc.float().mean()*100:.1f}%, "
          f"TS {m_c.ts.float().mean()*100:.1f}%, "
          f"EM {m_c.em.float().mean()*100:.1f}%, "
          f"cert {m_c.all_certified.float().mean()*100:.1f}%")

    cos_summary = {
        "rel": rel_c,
        "cert": m_c.all_certified.float().mean().item(),
        "ts": m_c.ts.float().mean().item(),
        "em": m_c.em.float().mean().item(),
    }
    del merged_cos, y_c, m_c
    gc.collect()

    # Cert-aware
    print("\nBuilding cert-aware plan (8 greedy steps)...")
    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001
    plan_ca = cert_aware_greedy_merge(
        moe, H, d_cos, target_n_clusters=target_n,
        max_bound=max_b, delta_em=0.1, bound_weight=0.0,
    )
    merged_ca = apply_merge_swiglu(moe, plan_ca, freq)
    with torch.no_grad():
        y_a, _ = merged_ca(H)
    rel_a = (y_o - y_a).abs().mean().item() / max(ref, 1e-8)
    m_a = check_conditions(H, moe.router.weight.detach(), plan_ca,
                           K=moe.cfg.top_k, delta_threshold=0.1)
    print(f"cert-aware: rel_div {rel_a*100:.1f}%, "
          f"MC {m_a.mc.float().mean()*100:.1f}%, "
          f"TS {m_a.ts.float().mean()*100:.1f}%, "
          f"EM {m_a.em.float().mean()*100:.1f}%, "
          f"cert {m_a.all_certified.float().mean()*100:.1f}%")

    print("\n=== Summary OLMoE layer 0, 64 → 56 ===")
    print(f"  cosine    : rel_div {cos_summary['rel']*100:.1f}%, "
          f"cert {cos_summary['cert']*100:.1f}%, "
          f"TS {cos_summary['ts']*100:.1f}%, "
          f"EM {cos_summary['em']*100:.1f}%")
    print(f"  cert-aware: rel_div {rel_a*100:.1f}%, "
          f"cert {m_a.all_certified.float().mean()*100:.1f}%, "
          f"TS {m_a.ts.float().mean()*100:.1f}%, "
          f"EM {m_a.em.float().mean()*100:.1f}%")


if __name__ == "__main__":
    main()
