"""Experiment 27 — Does the max-pool router merge help OLMoE (top-8)?

Exp 26 showed that for top-1 (Switch) the max-of-member-logits router merge
preserves routing EXACTLY, dissolving the expert↔routing tension (cosine +
max_logit dominated). The exact-preservation proof, however, is top-1 only.

For top-K the most max_logit can guarantee is COVERAGE: every cluster that
holds an original top-K expert is selected in the merged top-K. Proof: such a
cluster's logit = max(member logits) >= that top-K expert's logit > any
non-covering cluster's max member (which is ranked below the K-th expert);
since the original top-K experts collapse to m<=K clusters, all m are among
the K highest cluster logits. BUT when m < K the merged top-K activates
(K-m) EXTRA clusters that were not originally active, and the softmax gates
shift — so routing is NOT preserved exactly. This is exactly OLMoE's TS
fragility (top-8 of 64). This experiment quantifies how much max_logit buys.

Faithful OLMoE gating: norm_topk_prob=False, so gate = softmax over ALL
logits, select top-K entries WITHOUT renormalizing. lse_weight (what
apply_merge_swiglu does) is built to preserve total per-cluster probability
MASS under full softmax; max_logit is built to preserve top-K SELECTION.

Compares 3 router-merge schemes on freq-averaged cosine clusters, OLMoE
layer-0 cache (70 calib tokens, K=8), at 64->48 and 64->32:
  - lse_weight : cluster logit = H @ logsumexp(member router rows)
  - lse_logit  : cluster logit = logsumexp(member logits)
  - max_logit  : cluster logit = max(member logits)
Metrics: coverage% (orig top-K experts' clusters all in merged top-K),
exact-set TS% (merged top-K cluster set == covering set), rel_div.
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.merge import complete_linkage_clusters

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


def cosine_swiglu(W1, W2, W3):
    vecs = torch.stack([
        torch.cat([W1[i].flatten(), W2[i].flatten(), W3[i].flatten()])
        for i in range(W1.shape[0])
    ]).float()
    vecs = F.normalize(vecs, dim=-1)
    dist = 1.0 - vecs @ vecs.T
    dist.fill_diagonal_(0.0)
    return dist


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def cluster_experts(W1, W2, W3, plan, freq):
    out = []
    for members in plan.clusters:
        fr = freq[members]
        w = fr / fr.sum() if fr.sum() > 0 else torch.ones_like(fr) / len(fr)
        a1 = sum(w[i] * W1[m] for i, m in enumerate(members))
        a2 = sum(w[i] * W2[m] for i, m in enumerate(members))
        a3 = sum(w[i] * W3[m] for i, m in enumerate(members))
        out.append((a1, a2, a3))
    return out


@torch.no_grad()
def evaluate(H, W1, W2, W3, router_w, plan, freq, K, mode):
    B = H.shape[0]
    logits = H @ router_w.T                          # [B, N]
    n_clusters = len(plan.clusters)
    Kc = min(K, n_clusters)
    label_of = torch.tensor(plan.label_of)

    # --- original (faithful OLMoE: full softmax, top-K, no renorm) ---
    g_full = F.softmax(logits, dim=-1)
    orig_g, orig_idx = g_full.topk(K, dim=-1)        # [B,K]
    orig_clusters = label_of[orig_idx]               # [B,K] cluster of each

    # --- merged cluster logits per scheme ---
    if mode == "lse_weight":
        rows = torch.stack([torch.logsumexp(router_w[m], dim=0)
                            for m in plan.clusters], dim=0)
        cl = H @ rows.T                              # [B, n_clusters]
    elif mode == "lse_logit":
        cl = torch.stack([torch.logsumexp(logits[:, m], dim=-1)
                          for m in plan.clusters], dim=-1)
    elif mode == "max_logit":
        cl = torch.stack([logits[:, m].max(dim=-1).values
                          for m in plan.clusters], dim=-1)
    else:
        raise ValueError(mode)

    cg_full = F.softmax(cl, dim=-1)
    mg, msel = cg_full.topk(Kc, dim=-1)              # [B,Kc] merged top clusters

    # --- routing metrics ---
    # coverage: every original top-K expert's cluster is in merged top-Kc
    cover = torch.zeros(B)
    exact = torch.zeros(B)
    for b in range(B):
        oc = set(orig_clusters[b].tolist())          # covering clusters (<=K)
        ms = set(msel[b].tolist())
        cover[b] = 1.0 if oc.issubset(ms) else 0.0
        exact[b] = 1.0 if oc == ms else 0.0
    coverage = cover.mean().item()
    exact_ts = exact.mean().item()

    # --- output divergence (gated top-K SwiGLU) ---
    y_orig = torch.zeros(B, H.shape[1])
    for slot in range(K):
        idx = orig_idx[:, slot]
        g = orig_g[:, slot:slot+1]
        for e in idx.unique().tolist():
            m = idx == e
            y_orig[m] += g[m] * swiglu(H[m], W1[e], W2[e], W3[e])

    cexp = cluster_experts(W1, W2, W3, plan, freq)
    y_merged = torch.zeros_like(y_orig)
    for slot in range(Kc):
        idx = msel[:, slot]
        g = mg[:, slot:slot+1]
        for c in idx.unique().tolist():
            m = idx == c
            a1, a2, a3 = cexp[c]
            y_merged[m] += g[m] * swiglu(H[m], a1, a2, a3)

    rel = (y_orig - y_merged).abs().mean().item() / y_orig.abs().mean().item()
    return coverage, exact_ts, rel


def main():
    print(f"Loading OLMoE layer-0 cache from {CACHE}...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    router_w = c["router_weight"].float()
    H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    print(f"  N={N}, K={K}, H={tuple(H.shape)}, norm_topk_prob={c.get('norm_topk_prob')}")

    logits = H @ router_w.T
    _, topk_idx = logits.topk(K, dim=-1)
    freq = torch.zeros(N)
    for k in range(K):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))

    d_cos = cosine_swiglu(W1, W2, W3)

    for target_n in (48, 32):
        plan = complete_linkage_clusters(d_cos, find_cutoff(d_cos, target_n))
        print(f"\n{'='*72}")
        print(f"  OLMoE 64 -> {target_n}  (cosine clusters, K={K}, "
              f"{len(plan.clusters)} clusters)")
        print(f"{'='*72}")
        print(f"  {'router':12s} {'coverage':>10s} {'exact-TS':>10s} {'rel_div':>10s}")
        print("  " + "-" * 46)
        for mode in ("lse_weight", "lse_logit", "max_logit"):
            cov, ets, rel = evaluate(H, W1, W2, W3, router_w, plan, freq, K, mode)
            print(f"  {mode:12s} {cov*100:>9.1f}% {ets*100:>9.1f}% {rel*100:>9.2f}%")

    print(f"\n{'='*72}")
    print("  KEY: max_logit should give ~100% COVERAGE (top-K experts' clusters")
    print("  all selected) — the top-1 guarantee generalizes. But exact-TS and")
    print("  rel_div reveal the residual top-K cost: collapsed clusters force")
    print("  extra-cluster activation + gate shift that no router merge removes.")


if __name__ == "__main__":
    main()
