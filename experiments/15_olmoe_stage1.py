"""Experiment 15 — Stage 1 router stability on OLMoE-1B-7B (layer 0).

First real-SwiGLU validation. Goals:
  - Confirm the OLMoE adapter extracts router + experts correctly.
  - Compute the Stage 1 stability sweep.
  - Compare with Switch-base-8 (which had +13pp vs random init).

OLMoE-specific knobs:
  - 64 experts (vs Switch 8) — bigger N
  - top-K = 8 (vs Switch 1) — major change
  - norm_topk_prob = False — raw softmax probabilities for gating
  - SwiGLU, RoPE, decoder-only

Calibration is small (~10 short sentences) because the patched CPU
forward is slow. This is a sanity check, not a full benchmark.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.olmoe_adapter import (
    load_olmoe_block, collect_hidden_states_olmoe,
)
from cert_moe.router_stability import stable_topk_mask


SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models predict outputs from inputs.",
    "Neural networks consist of layers of neurons.",
    "Mixture of experts increases model capacity.",
    "Climate change is a pressing global concern.",
    "Education shapes future generations.",
    "Technology drives social change.",
    "Mathematics describes the universe.",
]


def main():
    print("Loading OLMoE layer 0 MoE block...")
    moe, model, tok = load_olmoe_block(layer_idx=0)

    print(f"\nMoE: d_model={moe.cfg.d_model}, d_ff={moe.cfg.d_ff}, "
          f"N={moe.cfg.n_experts}, top-K={moe.cfg.top_k}")
    print(f"Router weight shape: {tuple(moe.router.weight.shape)}, "
          f"dtype {moe.router.weight.dtype}")
    print(f"Expert 0 W1 shape: {tuple(moe.experts[0].W1.shape)}")
    print(f"Expert 0 W2 shape: {tuple(moe.experts[0].W2.shape)}")
    print(f"Expert 0 W3 shape: {tuple(moe.experts[0].W3.shape)}")

    print("\nCollecting calibration hidden states (slow due to patched "
          "fused MLP forward)...")
    H = collect_hidden_states_olmoe(
        model, tok, layer_idx=0, texts=SAMPLES, batch_size=2,
    )
    H = H.to(moe.router.weight.dtype)
    print(f"  Collected {H.shape[0]} tokens, d_model={H.shape[1]}")
    print(f"  ||h||_2 mean: {H.norm(dim=-1).mean().item():.3f}")

    if H.shape[0] > 512:
        H = H[torch.randperm(H.shape[0])[:512]]

    # Router logit stats
    with torch.no_grad():
        logits = moe.router(H)
    print(f"  Router logit std: {logits.std().item():.3f}")
    print(f"  Router L1/row mean: "
          f"{moe.router.weight.detach().abs().sum(-1).mean().item():.3f}")

    # Stage 1 sweep
    print(f"\n--- Stage 1 sweep ---")
    print(f"{'r':>8} | {'%top-K':>8} | {'%stable':>8} | "
          f"{'top-K→stable':>14}")
    print("-" * 50)
    for r in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        in_topk, stable = stable_topk_mask(
            H, moe.router.weight.detach(), r=r, K=moe.cfg.top_k,
        )
        pct_topk = in_topk.float().mean().item() * 100
        pct_stable = stable.float().mean().item() * 100
        frac = stable.float().sum().item() / max(
            in_topk.float().sum().item(), 1)
        print(f"{r:>8.4f} | {pct_topk:>7.2f}% | "
              f"{pct_stable:>7.2f}% | {frac*100:>13.2f}%")

    # Per-expert usage
    print(f"\n--- Per-expert usage at r=0.01 ---")
    _, stable = stable_topk_mask(
        H, moe.router.weight.detach(), r=0.01, K=moe.cfg.top_k,
    )
    with torch.no_grad():
        _, topk_idx = moe.router(H).topk(moe.cfg.top_k, dim=-1)
    usage = torch.zeros(moe.cfg.n_experts, dtype=torch.long)
    for k in range(moe.cfg.top_k):
        usage.scatter_add_(0, topk_idx[:, k],
                            torch.ones(H.shape[0], dtype=torch.long))
    stable_counts = stable.sum(0)
    print(f"  Expert usage: min {usage.min().item()}, "
          f"max {usage.max().item()}, "
          f"mean {usage.float().mean().item():.1f}")
    dump_ratio = usage.max().item() / max(usage.float().mean().item(), 1)
    print(f"  Dump ratio (max/mean): {dump_ratio:.2f}")
    # Top-5 most used experts
    top_used = usage.argsort(descending=True)[:5].tolist()
    print(f"\n  Top-5 most-used experts:")
    print(f"  {'expert':>6} | {'usage':>6} | {'stable':>7} | {'frac':>6}")
    for e in top_used:
        frac = stable_counts[e].item() / max(usage[e].item(), 1)
        print(f"  {e:>6} | {usage[e].item():>6} | "
              f"{stable_counts[e].item():>7} | {frac*100:>5.1f}%")


if __name__ == "__main__":
    main()
