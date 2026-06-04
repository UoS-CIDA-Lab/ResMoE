"""Experiment 33 — Baseline comparison (exp 17 setting) under the improved
router merge.

Reuses exp 17's EXACT setting: cached OLMoE layer 0, 32 calibration tokens,
the same competing compression methods (cosine, mc_smoe, hc_smoe, freq_prune,
cert-aware), the same ratios 64->48 / 64->32, the same rel_div + Theorem-1
cert% metrics. The ONLY addition: the router-merge OPERATOR is varied.

exp 17 used apply_merge_swiglu = lse_weight (log-sum-exp on router WEIGHT
rows). Exps 27/31 showed that operator is the worst; max_logit / lse_logit
roughly halve divergence and (exp 31) cut real PPL ~11x. The router-merge
operator is a drop-in improvement that applies to ANY clustering, so we apply
all three schemes to EVERY method and report rel_div, to fairly answer:

  - does the improved operator (max_logit) help every baseline? (it is a
    general contribution, not tied to "our" clustering)
  - under the best operator, how does our cert-aware stand vs the baselines?

The lse_weight column reproduces exp 17 (harness sanity check).
Gating matches exp 17's ToySwiGLU convention (renormalized softmax over the
selected top-K clusters).
"""
from __future__ import annotations

import sys
import pathlib
import math
import gc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cert_moe.toy_swiglu_moe import ToySwiGLUMoE, ToySwiGLUMoEConfig
from cert_moe.merge import cert_aware_greedy_merge, MergePlan, _build_label_of
from cert_moe.baselines import (
    cosine_distance_matrix, mc_smoe_distance_matrix,
    hc_smoe_distance_matrix, frequency_pruning_plan,
)
from cert_moe.theorem1_conditions import check_conditions

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


class LightSwiGLUExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W3 = nn.Parameter(torch.empty(d_model, d_ff))

    def forward(self, h):
        return (F.silu(h @ self.W1.T) * (h @ self.W2.T)) @ self.W3.T


def build_lean_moe(cache_path):
    cache = torch.load(cache_path, weights_only=True, map_location="cpu")
    cfg = ToySwiGLUMoEConfig(
        d_model=cache["d_model"], d_ff=cache["d_ff"],
        n_experts=cache["n_experts"], top_k=cache["top_k"], n_clone_pairs=0,
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
            e.W1.copy_(W1s[i]); e.W2.copy_(W2s[i]); e.W3.copy_(W3s[i])
        moe.experts.append(e)
    moe.clone_pairs = []
    del W1s, W2s, W3s
    gc.collect()
    return moe, H


def cluster_to_target(dist, target_n):
    """Single-pass complete-linkage to exactly target_n clusters (Lance-
    Williams); identical partition to exp 17's find_cutoff+complete_linkage."""
    D = dist.clone().float()
    D.fill_diagonal_(math.inf)
    clusters = [[i] for i in range(dist.shape[0])]
    while len(clusters) > target_n:
        idx = int(D.argmin().item())
        a, b = divmod(idx, D.shape[0])
        if a > b:
            a, b = b, a
        nr = torch.maximum(D[a], D[b])
        D[a] = nr; D[:, a] = nr; D[a, a] = math.inf
        keep = [i for i in range(D.shape[0]) if i != b]
        D = D[keep][:, keep]
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return MergePlan(clusters=clusters,
                     label_of=_build_label_of(clusters, dist.shape[0]))


def cluster_experts(moe, plan, freq):
    out = []
    for members in plan.clusters:
        fr = freq[members]
        w = fr / fr.sum() if fr.sum() > 0 else torch.ones_like(fr) / len(fr)
        W1 = sum(w[i] * moe.experts[m].W1 for i, m in enumerate(members))
        W2 = sum(w[i] * moe.experts[m].W2 for i, m in enumerate(members))
        W3 = sum(w[i] * moe.experts[m].W3 for i, m in enumerate(members))
        out.append((W1, W2, W3))
    return out


@torch.no_grad()
def rel_div(moe, plan, H, freq, scheme, y_o, ref, K):
    logits = H @ moe.router.weight.T                 # [B, 64]
    n_clusters = len(plan.clusters)
    Kc = min(K, n_clusters)
    if scheme == "lse_weight":
        rows = torch.stack([torch.logsumexp(moe.router.weight[m], dim=0)
                            for m in plan.clusters], dim=0)
        cl = H @ rows.T
    elif scheme == "lse_logit":
        cl = torch.stack([torch.logsumexp(logits[:, m], dim=1)
                          for m in plan.clusters], dim=1)
    elif scheme == "max_logit":
        cl = torch.stack([logits[:, m].max(dim=1).values
                          for m in plan.clusters], dim=1)
    else:
        raise ValueError(scheme)
    mv, mi = cl.topk(Kc, dim=-1)
    mg = F.softmax(mv, dim=-1)                        # renormalized (exp 17 conv)
    cexp = cluster_experts(moe, plan, freq)
    y_m = torch.zeros_like(y_o)
    for slot in range(Kc):
        idx = mi[:, slot]
        g = mg[:, slot:slot+1]
        for c in idx.unique().tolist():
            m = idx == c
            W1, W2, W3 = cexp[c]
            y_m[m] += g[m] * ((F.silu(H[m] @ W1.T) * (H[m] @ W2.T)) @ W3.T)
    return (y_o - y_m).abs().mean().item() / max(ref, 1e-8)


def main():
    print("Building lean MoE from cache (exp 17 setting)...")
    moe, H = build_lean_moe(str(CACHE))
    K = moe.cfg.top_k
    torch.manual_seed(0)
    if H.shape[0] > 32:
        H = H[torch.randperm(H.shape[0])[:32]]
    print(f"  N={moe.cfg.n_experts}, K={K}, calib H={tuple(H.shape)}")

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

    dmats = {
        "cosine": cosine_distance_matrix(moe.experts),
        "mc_smoe": mc_smoe_distance_matrix(
            moe.experts, H, moe.router.weight.detach(), K),
        "hc_smoe": hc_smoe_distance_matrix(moe.experts, H),
    }

    SCHEMES = ("lse_weight", "lse_logit", "max_logit")
    for target_n in (48, 32):
        print(f"\n{'='*74}")
        print(f"  Compression 64 -> {target_n}   "
              f"rel_div under each router merge (lower=better)")
        print(f"{'='*74}")
        # build each method's plan
        plans = {}
        for name, d in dmats.items():
            plans[name] = cluster_to_target(d, target_n)
        plans["freq_prune"] = frequency_pruning_plan(
            moe.experts, H, moe.router.weight.detach(), K, target_n)
        d_gate = dmats["cosine"]
        plans["cert-aware"] = cert_aware_greedy_merge(
            moe, H, d_gate, target_n_clusters=target_n,
            max_bound=d_gate[d_gate.isfinite()].max().item() * 1.001,
            delta_em=0.1, bound_weight=0.0)

        print(f"  {'method':12s} "
              f"{'lse_weight':>11s} {'lse_logit':>11s} {'max_logit':>11s} "
              f"{'cert%':>7s}")
        print("  " + "-" * 58)
        for name, plan in plans.items():
            rels = {s: rel_div(moe, plan, H, freq, s, y_o, ref, K)
                    for s in SCHEMES}
            cert = check_conditions(
                H, moe.router.weight.detach(), plan, K=K,
                delta_threshold=0.1).all_certified.float().mean().item()
            tag = name + (" *" if name == "cert-aware" else "")
            print(f"  {tag:12s} "
                  f"{rels['lse_weight']*100:>10.1f}% "
                  f"{rels['lse_logit']*100:>10.1f}% "
                  f"{rels['max_logit']*100:>10.1f}% "
                  f"{cert*100:>6.1f}%")

    print(f"\n{'='*74}")
    print("  lse_weight column = exp 17 (sanity). max_logit = improved operator.")
    print("  Honest read: which (clustering x operator) wins, and does the")
    print("  operator help the BASELINES too (general contribution)?")


if __name__ == "__main__":
    main()
