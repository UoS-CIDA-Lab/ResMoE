"""Experiment 44 — Certified aliasing + per-input eps-ball certification on
OLMoE SwiGLU (top-8). Extension ① of the exp 40-43 line from Switch to OLMoE.

The structural argument generalizes to top-K: the aliased model and original
SHARE the router, select the SAME top-8, and differ only in aliased experts'
functions; deviation = sum over selected-and-aliased experts of gate*alias_error;
routing flips together (no unbounded error). Within the top-8 SET-stable radius,
the deviation over an eps-ball is bounded by the aliased experts' SwiGLU diff
bound over the small box [h-eps, h+eps].

Questions: (1) does the per-input small-eps-ball make the SwiGLU equivalence
bound TIGHT (swiglu_bounds were ~10000x loose globally per olmoe-port-status)?
(2) what is the top-8 SET-stable radius (expect small — TS fragility)?

Uses the OLMoE layer-0 cache. Run: python3 experiments/44_olmoe_certified_aliasing.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import swiglu_diff_bound_box

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def emp_pair_diff(W1i, W2i, W3i, W1j, W2j, W3j, Hs):
    if Hs.shape[0] == 0:
        return float("inf")
    with torch.no_grad():
        d = (swiglu(Hs, W1i, W2i, W3i) - swiglu(Hs, W1j, W2j, W3j)).abs()
    return d.max(-1).values.max().item()


def pgd_pair_diff(Wi, Wj, h0, eps, n_steps=100):
    W1i, W2i, W3i = Wi
    W1j, W2j, W3j = Wj
    step = eps / 10
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps):
        hg = h.clone().requires_grad_(True)
        d = (swiglu(hg, W1i, W2i, W3i) - swiglu(hg, W1j, W2j, W3j)
             ).abs().max(-1).values.sum()
        d.backward()
        with torch.no_grad():
            h = torch.max(torch.min(h + step * hg.grad.sign(), h0 + eps), h0 - eps)
            cur = (swiglu(h, W1i, W2i, W3i) - swiglu(h, W1j, W2j, W3j)
                   ).abs().max(-1).values.max().item()
            best = max(best, cur)
    return best


def topk_set_stable_radius(logits_row, W_g, K):
    """Largest L_inf r keeping the top-K SET fixed:
    min over (i in topK, k not in topK) of (l_i - l_k)/||W_i - W_k||_1."""
    vals, idx = logits_row.topk(K)
    inset = set(idx.tolist())
    out = [k for k in range(W_g.shape[0]) if k not in inset]
    r = float("inf")
    for i in inset:
        for k in out:
            denom = (W_g[i] - W_g[k]).abs().sum().item()
            if denom > 0:
                r = min(r, (logits_row[i] - logits_row[k]).item() / denom)
    return r


def main():
    print(f"Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float()
    H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    print(f"  N={N}, K={K}, H={tuple(H.shape)}, ||h|| mean {H.norm(dim=-1).mean():.1f}")

    logits = H @ Wg.T
    _, topk_idx = logits.topk(K, dim=-1)
    freq = torch.zeros(N)
    for k in range(K):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool)
    inset.scatter_(1, topk_idx, True)

    # alias 2 lowest-freq experts WITH freq>0 (need routed inputs for per-input
    # certification) -> empirical nearest twin over that expert's routed region
    used = [i for i in freq.argsort().tolist() if freq[i] > 0]
    aliased = used[:2]
    rep_of = list(range(N))
    diffs = []
    for i in aliased:
        routed_i = H[inset[:, i]]
        best = None
        for j in range(N):
            if j == i:
                continue
            d = emp_pair_diff(W1[i], W2[i], W3[i], W1[j], W2[j], W3[j], routed_i)
            if best is None or d < best[0]:
                best = (d, j)
        rep_of[i] = best[1]
        diffs.append(round(best[0], 2))
    print(f"  aliased {aliased} (freq {[int(freq[i]) for i in aliased]}) -> "
          f"{[rep_of[i] for i in aliased]}  emp routed diff {diffs}")

    # top-8 SET-stable radius distribution
    rr = torch.tensor([topk_set_stable_radius(logits[b], Wg, K)
                       for b in range(H.shape[0])])
    rr = rr[torch.isfinite(rr)]
    qs = torch.quantile(rr, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    print(f"\n  top-{K} SET-stable radius r_route (L_inf): median {rr.median():.4f}, "
          f"q[10,25,50,75,90]={[round(x,4) for x in qs.tolist()]}")

    # per-input eps-ball: SwiGLU diff bound vs PGD (for aliased experts)
    # NOTE: set-stability is NOT required for the aliased-vs-original deviation,
    # because both models SHARE the router (flip together); the deviation is
    # bounded by the aliased experts' function diff over the ball (gate<=1).
    print(f"\n{'='*70}")
    print("  per-input eps-ball SwiGLU alias-error bound: CROWN vs PGD")
    print("  (no routing-stability needed: shared router)")
    print(f"{'='*70}")
    print(f"  {'eps':>6s} {'med cert':>10s} {'med emp':>10s} {'looseness':>10s}")
    print("  " + "-" * 44)
    pool = torch.cat([torch.nonzero(inset[:, i]).flatten() for i in aliased])
    torch.manual_seed(0)
    pool = pool[torch.randperm(len(pool))[:20]]
    for eps in (0.01, 0.025, 0.05, 0.1):
        certs, emps = [], []
        for b in pool.tolist():
            h = H[b:b + 1]
            i = next((a for a in aliased if inset[b, a]), None)
            if i is None:
                continue
            j = rep_of[i]
            cert = swiglu_diff_bound_box(
                W1[i], W2[i], W3[i], W1[j], W2[j], W3[j],
                (h - eps).squeeze(0), (h + eps).squeeze(0))
            emp = pgd_pair_diff((W1[i], W2[i], W3[i]),
                                (W1[j], W2[j], W3[j]), h, eps)
            certs.append(float(cert)); emps.append(emp)
        mc = sorted(certs)[len(certs) // 2]
        me = sorted(emps)[len(emps) // 2]
        print(f"  {eps:>6.3f} {mc:>10.4f} {me:>10.4f} {mc/max(me,1e-6):>9.1f}x")

    print(f"\n{'='*70}")
    print("  Switch (exp43) was 1.4x@0.1. SwiGLU on a small ball should also be")
    print("  tight. The tiny top-8 SET-stable radius (TS fragility) does NOT")
    print("  limit aliased-vs-original equivalence (shared router) — it would")
    print("  only matter if we cared which experts fire.")


if __name__ == "__main__":
    main()
