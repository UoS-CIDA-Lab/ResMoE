"""Experiment 01 — Sanity check the closed-form router stability test.

Questions answered:
  1. At r = 0, is every top-K assignment "stable"? (trivially yes; sanity)
  2. As r grows, what fraction of (token, expert) pairs remain stable?
  3. How does stability vary across experts? (some experts may be
     "fragile" — their top-K membership flips under tiny perturbation.)

This is Stage 1 of the CertMerge framework. If the closed-form check is
too strict (almost nothing stable at meaningful r) or too loose (every-
thing stable even at huge r) the framework is in trouble.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.toy_moe import ToyMoE, ToyMoEConfig, make_calibration_data
from cert_moe.router_stability import stable_topk_mask


def main():
    torch.manual_seed(0)
    cfg = ToyMoEConfig(d_model=64, d_ff=128, n_experts=16, top_k=2,
                       n_clone_pairs=2)
    moe = ToyMoE(cfg)
    H = make_calibration_data(moe, n_samples=512)

    print(f"Toy MoE: N={cfg.n_experts}, top-K={cfg.top_k}, d={cfg.d_model}")
    print(f"Planted clone pairs: {moe.clone_pairs}")
    print(f"Calibration samples: {H.shape[0]}\n")

    # Estimate the "natural scale" of router logits for choosing r values
    with torch.no_grad():
        logits = H @ moe.router.weight.T
    logit_scale = logits.std().item()
    print(f"Router logit std: {logit_scale:.3f}")
    print(f"Router weight L1-per-row: "
          f"{moe.router.weight.detach().abs().sum(-1).mean().item():.3f}\n")

    # Sweep r over a sensible range
    r_values = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"{'r':>8} | {'%top-K':>8} | {'%stable':>8} | {'%top-K → stable':>16}")
    print("-" * 56)
    for r in r_values:
        in_topk, stable = stable_topk_mask(
            H, moe.router.weight.detach(), r=r, K=cfg.top_k
        )
        pct_topk = in_topk.float().mean().item() * 100
        pct_stable = stable.float().mean().item() * 100
        # Among (b, i) in top-K, what fraction are stable?
        frac_topk_stable = stable.float().sum().item() / max(in_topk.float().sum().item(), 1)
        print(f"{r:>8.4f} | {pct_topk:>7.2f}% | {pct_stable:>7.2f}% | {frac_topk_stable*100:>15.2f}%")

    # Per-expert stability at a chosen r
    r_focus = 0.01
    print(f"\nPer-expert stability at r={r_focus}:")
    _, stable = stable_topk_mask(H, moe.router.weight.detach(), r=r_focus, K=cfg.top_k)
    counts = stable.sum(0)
    in_topk_counts = (H @ moe.router.weight.T).topk(cfg.top_k, dim=-1).indices
    topk_usage = torch.zeros(cfg.n_experts, dtype=torch.long)
    for k in range(cfg.top_k):
        topk_usage.scatter_add_(0, in_topk_counts[:, k],
                                torch.ones(H.shape[0], dtype=torch.long))
    print(f"{'expert':>6} | {'top-K hits':>10} | {'stable':>8} | {'stable frac':>12}")
    print("-" * 50)
    for i in range(cfg.n_experts):
        is_planted = any(i in p for p in moe.clone_pairs)
        tag = "  *clone" if is_planted else ""
        frac = counts[i].item() / max(topk_usage[i].item(), 1)
        print(f"{i:>6} | {topk_usage[i].item():>10} | {counts[i].item():>8} | {frac*100:>11.1f}%{tag}")


if __name__ == "__main__":
    main()
