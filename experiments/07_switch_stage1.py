"""Experiment 07 — Stage 1 router stability on Switch-base-8.

Repeat the toy stability sweep (experiment 01) on a real, pretrained
MoE layer. Sanity check: closed-form check should still produce a
monotonically decreasing stable% curve as r grows.

If the curve looks similar to toy → framework is portable.
If it's very different → real router has structural properties we
need to handle.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.router_stability import stable_topk_mask


def main():
    print("Loading Switch-base-8 first MoE block (encoder.block.1.layer.1.mlp)...")
    moe, model, tok = load_switch_block()

    print(f"\nMoE block: d_model={moe.cfg.d_model}, "
          f"d_ff={moe.cfg.d_ff}, "
          f"N={moe.cfg.n_experts}, top-K={moe.cfg.top_k}")

    # Collect calibration hidden states via short WikiText-style sentences
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago—never mind how long precisely.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "The cat sat on the mat.",
        "Machine learning models predict outputs from inputs.",
        "Neural networks consist of layers of neurons.",
        "Transformers use attention mechanisms.",
        "Mixture of experts increases model capacity.",
        "Quantum computers may revolutionize computation.",
        "Climate change is a pressing global concern.",
        "Economic policy affects everyone.",
        "Literature reflects the human condition.",
        "Mathematics describes the universe.",
        "Music transcends language barriers.",
    ] * 8   # repeat for more tokens (~128 sentences)
    print(f"\nCollecting hidden states from {len(sample_texts)} sentences...")
    H = collect_hidden_states(
        model, tok, "encoder.block.1.layer.1.mlp",
        sample_texts, batch_size=8,
    )
    print(f"Collected {H.shape[0]} tokens, d_model={H.shape[1]} "
          f"(dtype {H.dtype})")

    # Cast to router dtype to avoid mismatch (Switch keeps router in float32
    # while activations might be bf16).
    H = H.to(moe.router.weight.dtype)

    # Subsample for efficiency
    if H.shape[0] > 2048:
        idx = torch.randperm(H.shape[0])[:2048]
        H = H[idx]
        print(f"Subsampled to {H.shape[0]} tokens, cast to {H.dtype}")

    # Compute router logit statistics
    with torch.no_grad():
        logits = moe.router(H)
    logit_std = logits.std().item()
    router_w_l1 = moe.router.weight.detach().abs().sum(-1).mean().item()
    print(f"\nReal router stats:")
    print(f"  Logit std        : {logit_std:.3f}")
    print(f"  Router L1/row    : {router_w_l1:.3f}")
    print(f"  Hidden-state norm: mean ||h||_2 = "
          f"{H.norm(dim=-1).mean().item():.3f}")

    # Stage 1 sweep
    print(f"\n{'r':>10} | {'%top-K':>8} | {'%stable':>8} | "
          f"{'%top-K → stable':>16}")
    print("-" * 60)
    for r in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        in_topk, stable = stable_topk_mask(
            H, moe.router.weight.detach(), r=r, K=moe.cfg.top_k
        )
        pct_topk = in_topk.float().mean().item() * 100
        pct_stable = stable.float().mean().item() * 100
        frac = stable.float().sum().item() / max(in_topk.float().sum().item(), 1)
        print(f"{r:>10.4f} | {pct_topk:>7.2f}% | "
              f"{pct_stable:>7.2f}% | {frac*100:>15.2f}%")

    # Per-expert stability at r=0.05
    print(f"\nPer-expert usage and stability at r=0.05:")
    r_focus = 0.05
    _, stable = stable_topk_mask(
        H, moe.router.weight.detach(), r=r_focus, K=moe.cfg.top_k
    )
    with torch.no_grad():
        topk_idx = moe.router(H).topk(moe.cfg.top_k, dim=-1).indices
    usage = torch.zeros(moe.cfg.n_experts, dtype=torch.long)
    for k in range(moe.cfg.top_k):
        usage.scatter_add_(0, topk_idx[:, k],
                           torch.ones(H.shape[0], dtype=torch.long))
    stable_counts = stable.sum(0)
    print(f"  {'expert':>6} | {'usage':>6} | {'stable':>7} | {'stable frac':>12}")
    print("  " + "-" * 50)
    for i in range(moe.cfg.n_experts):
        frac = stable_counts[i].item() / max(usage[i].item(), 1)
        print(f"  {i:>6} | {usage[i].item():>6} | "
              f"{stable_counts[i].item():>7} | {frac*100:>11.1f}%")


if __name__ == "__main__":
    main()
