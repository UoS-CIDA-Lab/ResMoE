"""Experiment 85 — Is the active top-K set REDUNDANT (precomputable as a verified
asset), beyond router confidence?

User's idea: instead of router confidence, decide compute reduction from REDUNDANCY
among the selected experts, precomputed offline as a pairwise/set-wise asset. Two
distinct notions (exp 46 only tested global pairwise twins -> none):
  (1) CO-ACTIVATION pairwise: when experts i,j are BOTH in top-K (so h is in their
      joint region), is E_i(h) ~ E_j(h)? (conditional redundancy, region-specific.)
  (2) SET-WISE low rank: even if experts are individually distinct, does the GATED
      output set {g_e E_e(h)}_{e in topK} have effective rank r < K? i.e. do a few
      experts' contributions span the layer output, so the rest are reconstructible?
      This is NOT pairwise similarity; it asks if the active set is collinear/low-rank.
We measure both on real routed data, plus the implied compute saving and error.

CPU, OLMoE layer-0 cache. Run: python3 experiments/85_topk_redundancy_asset.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def main():
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float(); H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    T = H.shape[0]
    logits = H @ Wg.T
    probs = F.softmax(logits, dim=1)
    topv, topi = probs.topk(K, dim=-1)            # gates (over all N), indices
    print(f"  {T} tokens, N={N}, K={K}", flush=True)

    # precompute each selected expert output per token
    # E[t, slot] = E_{topi[t,slot]}(H[t]); gE = gate * E
    E = torch.zeros(T, K, W3.shape[1])
    for t in range(T):
        for s in range(K):
            e = topi[t, s].item()
            E[t, s] = swiglu(H[t:t+1], W1[e], W2[e], W3[e]).squeeze(0)
    gE = topv.unsqueeze(-1) * E                    # [T,K,d] gated contributions
    Y = gE.sum(1)                                  # [T,d] layer output
    ynorm = Y.norm(dim=1)                          # [T]

    # ---- (2) SET-WISE effective rank of the gated contributions ----
    print(f"\n{'='*70}\n  (2) SET-WISE: effective rank of the K gated contributions "
          f"{{g_e E_e}}\n{'='*70}")
    print(f"  how many of the K={K} contributions are needed to reconstruct the gated")
    print(f"  output to a relative tolerance (energy of top-r singular values):")
    print(f"  {'rel tol':>8s} | {'avg rank r needed':>17s} | {'compute saving (K-r)/K':>22s}")
    print("  " + "-"*54)
    for tol in [0.01, 0.02, 0.05, 0.10]:
        ranks = []
        for t in range(T):
            M = gE[t]                              # [K, d]
            s = torch.linalg.svdvals(M)            # singular values
            energy = (s ** 2).cumsum(0) / (s ** 2).sum().clamp(min=1e-12)
            r = int((energy < (1 - tol)).sum().item()) + 1
            ranks.append(min(r, K))
        ar = sum(ranks) / len(ranks)
        print(f"  {tol:>8.2f} | {ar:>17.2f} | {(K-ar)/K*100:>21.1f}%", flush=True)
    print("  (NOTE: low rank here means the gated outputs are COLLINEAR -- a few")
    print("   directions span them -- which would let a few experts reconstruct the rest.)")

    # ---- (1) CO-ACTIVATION pairwise redundancy ----
    print(f"\n{'='*70}\n  (1) CO-ACTIVATION pairwise: ||E_i - E_j|| / ||output|| when i,j "
          f"co-selected\n{'='*70}")
    # for each co-selected pair at each token, relative distance
    rels = []
    for t in range(T):
        for a in range(K):
            for b in range(a + 1, K):
                d = (E[t, a] - E[t, b]).norm().item()
                rels.append(d / max(ynorm[t].item(), 1e-9))
    rels = torch.tensor(rels)
    print(f"  co-selected expert pairs: {len(rels)}")
    print(f"  ||E_i-E_j|| / ||Y||  : min {rels.min():.2f}, median {rels.median():.2f}, "
          f"max {rels.max():.2f}")
    for d in [0.1, 0.25, 0.5, 1.0]:
        print(f"    pairs with rel-dist <= {d}: {int((rels <= d).sum().item())}/{len(rels)} "
              f"({(rels <= d).float().mean()*100:.1f}%)", flush=True)
    print("  (pairwise twins ~ none if these are large -- consistent with exp 46;")
    print("   the SET-WISE rank above is the independent, possibly-more-favorable signal.)")

    # ---- implication: a few experts dominate the gated output? ----
    print(f"\n{'='*70}\n  Contribution concentration (is the output dominated by few experts?)"
          f"\n{'='*70}")
    contrib = gE.norm(dim=2)                        # [T,K] per-expert gated magnitude
    frac = contrib / contrib.sum(1, keepdim=True)
    sortc = frac.sort(1, descending=True).values
    cum = sortc.cumsum(1).mean(0)                   # avg cumulative contribution mass
    print("  avg cumulative contribution mass by #experts (sorted desc):")
    print("    " + "  ".join(f"{i+1}:{cum[i]*100:.0f}%" for i in range(K)), flush=True)
    print("  if the first few already ~100%, the rest contribute little (skippable);")
    print("  but this is the same gate*||E|| signal as exp 82/84 (needs the outputs).")


if __name__ == "__main__":
    main()
