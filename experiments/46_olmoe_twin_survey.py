"""Experiment 46 — Are there genuinely equivalent expert twins? (extension ③)

The whole certified-aliasing thesis only buys real (near-lossless) compression
if experts actually have near-equivalent twins. exp 41/24 found experts are
distinct in WEIGHT space (cosine ~0.98). But aliasing error is OUTPUT distance,
not weight distance — two differently-parameterized experts can still be output-
equivalent. exp 44 saw low-freq OLMoE experts with tiny twin distance (0.02).
This surveys ALL experts: per-expert nearest OUTPUT twin distance (over the
calibration tokens), normalized by the expert's own output magnitude, and the
aliasing budget curve (#aliasable vs relative-delta tolerance).

OLMoE layer-0 cache, 64 experts. Output |y| is small (~0.03), so we report
RELATIVE twin distance (twin / own output scale). Run:
  python3 experiments/46_olmoe_twin_survey.py
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
    print("Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float()
    H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    logits = H @ Wg.T
    _, topk_idx = logits.topk(K, dim=-1)
    freq = torch.zeros(N)
    for k in range(K):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    print(f"  N={N}, K={K}, {H.shape[0]} tokens")

    # all-expert outputs on the calib tokens
    with torch.no_grad():
        Y = torch.stack([swiglu(H, W1[i], W2[i], W3[i]) for i in range(N)])  # [N,B,d]
    out_scale = Y.abs().amax(dim=-1).amax(dim=-1)            # [N] per-expert |y|_inf
    # pairwise output distance D[i,j] = max over tokens ||E_i - E_j||_inf
    D = torch.full((N, N), float("inf"))
    for i in range(N):
        diff = (Y[i].unsqueeze(0) - Y).abs().amax(dim=-1).amax(dim=-1)  # [N]
        diff[i] = float("inf")
        D[i] = diff

    # per-expert nearest twin (absolute and relative)
    nearest = D.min(dim=1)
    rel = nearest.values / out_scale.clamp(min=1e-8)
    order = rel.argsort()
    print(f"\n  per-expert output scale |y|_inf: median {out_scale.median():.4f}")
    print(f"  nearest-twin RELATIVE distance (twin / own |y|): "
          f"median {rel.median():.3f}, min {rel.min():.3f}, max {rel.max():.3f}")
    print("\n  10 most twin-able experts (smallest relative twin distance):")
    print(f"  {'expert':>7s} {'twin':>5s} {'freq':>5s} {'abs':>9s} {'rel':>7s}")
    for i in order[:10].tolist():
        j = int(nearest.indices[i])
        print(f"  {('E'+str(i)):>7s} {('E'+str(j)):>5s} {int(freq[i]):>5d} "
              f"{nearest.values[i]:>9.4f} {rel[i]:>6.2f}")

    # aliasing budget curve: greedy alias (low-freq first) to a kept rep within
    # relative tolerance; count + memory saving (experts are 93% of model params)
    print(f"\n{'='*60}\n  Aliasing budget vs relative-delta tolerance\n{'='*60}")
    print(f"  {'rel_tol':>8s} {'#aliasable':>11s} {'expert-mem saved':>17s}")
    for tol in (0.05, 0.1, 0.25, 0.5, 1.0):
        rep_is = [True] * N
        aliased = []
        for i in freq.argsort().tolist():
            cands = [(D[i, j].item(), j) for j in range(N)
                     if j != i and rep_is[j] and j not in aliased]
            cands = [(d, j) for d, j in cands
                     if d <= tol * out_scale[i].item()]
            if cands:
                rep_is[i] = False
                aliased.append(i)
        print(f"  {tol:>8.2f} {len(aliased):>11d} {len(aliased)/N*100:>15.1f}%")

    print(f"\n{'='*60}")
    print("  rel twin distance << 1 => genuinely output-equivalent twins exist")
    print("  (near-lossless aliasing). If most experts have rel ~ 1, they are")
    print("  distinct and aliasing is inherently lossy. NOTE: only 70 calib")
    print("  tokens / layer 0 — a live-model, multi-layer survey would confirm.")


if __name__ == "__main__":
    main()
