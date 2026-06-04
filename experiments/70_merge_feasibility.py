"""Experiment 70 — Feasibility probe for CERTIFIED expert merging (scenario 2).

Before building a method, check the premise: do certifiably mergeable expert PAIRS
exist on OLMoE? Merging i->j (route i's tokens to j) changes the output by
||E_i(x) - E_j(x)|| for x in i's operating region. It is certified-safe at tolerance
delta iff that is provably <= delta over the region. We measure:
  (1) empirical nearest-neighbour output distance per expert (max over its routed
      inputs of min_j ||E_i - E_j||_inf), normalised by output magnitude;
  (2) how many experts have SOME merge partner within delta (mergeable count vs delta);
  (3) for the closest pairs, the CERTIFIED bound (exp-54 difference bound applied to
      two distinct experts) -- is it tight enough to certify the merge?
If (2) is ~0 at usable delta, certified pairwise merging is infeasible on OLMoE (its
64 fine-grained experts are deliberately diverse, cf. exp 46) and scenario 2 must
move to a redundant model or a different merge form (centroid+residual, sharing).

CPU, OLMoE layer-0 cache. Run: python3 experiments/70_merge_feasibility.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
EPS = 0.05


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def silu_grad(z):
    s = torch.sigmoid(z); return s + z * s * (1 - s)


def affine_box(W, lo, hi):
    c = (lo + hi) / 2; r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_consts():
    z = torch.linspace(-30, 30, 1200001).requires_grad_(True)
    g1, = torch.autograd.grad(silu(z).sum(), z, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    return (g1.abs().max().item()*1.01, g2.abs().max().item()*1.01, g3.abs().max().item()*1.01)


def bigspec(W):
    return torch.linalg.matrix_norm(W, ord=2).item()


def cert_diff(W1, W2, W3, W1b, W2b, W3b, h0, eps, Ls, Lpp, Lppp):
    """Certified bound on ||E_a(x) - E_b(x)||_inf over B2(h0,eps) for two experts a,b."""
    s1, s2 = bigspec(W1), bigspec(W2)
    s1q, s2q = bigspec(W1b), bigspec(W2b)
    sd1, sd2 = bigspec(W1 - W1b), bigspec(W2 - W2b)
    d1, d2, d3 = W1 - W1b, W2 - W2b, W3 - W3b
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi)); Bvq = amax(*affine_box(W2b, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi)); Bd2 = amax(*affine_box(d2, lo, hi))
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)
    De = Ls * W3.abs().amax(1)
    dDc = (W3.abs()*(Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0) + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs()*(Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    HD = Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JD = (J(W1, W2, W3) - J(W1b, W2b, W3b)).norm(dim=1)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1b, W2b, W3b)).squeeze(0).abs()
    return (Dh0 + eps*JD + 0.5*eps*eps*HD).max().item()


def main():
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float(); H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    Ls, Lpp, Lppp = silu_consts()
    topk = (H @ Wg.T).topk(K, dim=-1).indices
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool); inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist() for e in active}

    # E_e(h_t) for all active experts on ALL tokens
    outs = {e: swiglu(H, W1[e], W2[e], W3[e]) for e in active}     # [T, d]
    omag = torch.stack([outs[e].abs().amax(1).mean() for e in active]).mean().item()
    print(f"  {len(active)} active experts; mean |E(h)|_inf ~ {omag:.4f}", flush=True)

    # empirical nearest-neighbour over each expert's routed inputs
    nn = {}
    for i in active:
        ti = routed[i]
        best, bj = 1e9, None
        for j in active:
            if j == i:
                continue
            d = max((outs[i][t] - outs[j][t]).abs().max().item() for t in ti)
            if d < best:
                best, bj = d, j
        nn[i] = (best, bj)
    dists = torch.tensor([nn[i][0] for i in active])
    rel = dists / omag
    print(f"\n  Empirical nearest-twin output distance (||E_i - E_j||_inf over i's region):")
    print(f"    abs : mean {dists.mean():.4f} median {dists.median():.4f} min {dists.min():.4f}")
    print(f"    rel to |E| : mean {rel.mean():.2f} median {rel.median():.2f} "
          f"min {rel.min():.2f}  (>=1 means twin differs by ~full output magnitude)")

    print(f"\n  Mergeable experts (have a partner within delta, EMPIRICAL) vs delta:")
    for d in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        cnt = sum(nn[i][0] <= d for i in active)
        print(f"    delta={d:>5.2f}: {cnt:>3d}/{len(active)} experts "
              f"({cnt/len(active)*100:>4.0f}%)  -> up to {cnt} merges, "
              f"{cnt/2/len(active)*100:.0f}% expert reduction (paired)")

    # certified bound for the 5 closest pairs (does the cert clear the empirical dist?)
    print(f"\n  Certified bound vs empirical for the 5 closest pairs (eps_L2={EPS}):")
    order = sorted(active, key=lambda i: nn[i][0])[:5]
    for i in order:
        j = nn[i][1]; t = routed[i][0]
        cb = cert_diff(W1[i], W2[i], W3[i], W1[j], W2[j], W3[j], H[t:t+1],
                       EPS, Ls, Lpp, Lppp)
        print(f"    {i:>2d}->{j:<2d}: empirical {nn[i][0]:.4f} (rel {nn[i][0]/omag:.2f}), "
              f"certified {cb:.4f}", flush=True)
    print(f"\n  If even the EMPIRICAL nearest-twin distance is ~|E| (rel ~1), no pair is")
    print("  close, so certified pairwise merging is infeasible here regardless of bound")
    print("  tightness -> scenario 2 needs a redundant model or a different merge form.")


if __name__ == "__main__":
    main()
