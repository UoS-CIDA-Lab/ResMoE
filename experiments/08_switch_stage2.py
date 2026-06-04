"""Experiment 08 — Stage 2 expert pair bounds on Switch-base-8.

Critical question: in toy MoE (random init) our CROWN bound was 100-300×
looser than empirical. Does PRETRAINED Switch-base-8 give tighter bounds?

Why we might expect tighter:
  - Real experts develop low-rank-ish structure during training (some
    redundancy → some pairs are functionally close).
  - Hidden state distribution is much more concentrated than uniform box.
  - With zonotope input region from calibration, bound should shrink.

Why we might expect looser:
  - d_ff = 3072 is 24× bigger than toy (128). Looseness compounds.
  - Real experts likely have larger weight magnitudes.

We compute pairwise bounds for all C(8,2) = 28 pairs and compare:
  - Empirical max ||E_i - E_j||_∞ on calibration
  - α-CROWN box bound
  - α-CROWN zonotope bound
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from itertools import combinations

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import (
    pair_diff_bound, empirical_pair_diff,
    zonotope_from_data, alpha_crown_zonotope_bound,
)


def main():
    print("Loading Switch-base-8 first MoE block...")
    moe, model, tok = load_switch_block()

    print(f"Block: d_model={moe.cfg.d_model}, d_ff={moe.cfg.d_ff}, "
          f"N={moe.cfg.n_experts}, top-K={moe.cfg.top_k}")

    # Calibration — same as Stage 1
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago—never mind how long precisely.",
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
        "Education shapes future generations.",
        "Technology drives social change.",
    ] * 6
    print(f"\nCollecting hidden states from {len(sample_texts)} sentences...")
    H = collect_hidden_states(
        model, tok, "encoder.block.1.layer.1.mlp",
        sample_texts, batch_size=8,
    )
    H = H.to(moe.router.weight.dtype)
    if H.shape[0] > 1024:
        idx = torch.randperm(H.shape[0])[:1024]
        H = H[idx]
    print(f"Calibration set: {H.shape} (dtype {H.dtype})")

    # Hidden-state statistics — important for bound interpretation
    print(f"\nHidden state stats:")
    print(f"  ||h||_2 mean : {H.norm(dim=-1).mean().item():.3f}")
    print(f"  per-dim std : {H.std(0).mean().item():.3f}")
    print(f"  per-dim range max : "
          f"{(H.max(0).values - H.min(0).values).max().item():.3f}")

    # Box input region
    h_lo = H.min(0).values - 0.05
    h_hi = H.max(0).values + 0.05

    # Zonotope input region — try several PCA dimensions
    print(f"\nBuilding zonotope (k=64 PCA components)...")
    center, V, alpha_coef = zonotope_from_data(H, n_components=64, margin=0.05)

    # All pairs
    pairs = list(combinations(range(moe.cfg.n_experts), 2))
    print(f"\nComputing bounds for {len(pairs)} expert pairs...")
    print(f"{'pair':>6} | {'empirical':>10} | {'α-CROWN (box)':>13} | "
          f"{'α-CROWN (zono k=64)':>20} | {'ratio (box/emp)':>14}")
    print("-" * 90)

    results = []
    for k, (i, j) in enumerate(pairs):
        emp = empirical_pair_diff(moe.experts[i], moe.experts[j], H)
        b_box = pair_diff_bound(
            moe.experts[i], moe.experts[j], h_lo, h_hi,
            method="alpha_crown", alpha_iters=15,
        )
        b_zono = alpha_crown_zonotope_bound(
            moe.experts[i], moe.experts[j], center, V, alpha_coef,
            n_iters=15,
        )
        ratio = b_box / max(emp, 1e-8)
        results.append((i, j, emp, b_box, b_zono, ratio))
        print(f"  ({i},{j}) | {emp:>10.3f} | {b_box:>13.3f} | "
              f"{b_zono:>20.3f} | {ratio:>13.1f}x")
        print(f"        [{k+1}/{len(pairs)}]", end="\r")

    # Summary
    emp_vals = [r[2] for r in results]
    box_vals = [r[3] for r in results]
    zono_vals = [r[4] for r in results]
    print(f"\n\n--- Summary across all 28 pairs ---")
    print(f"Empirical max diff: min={min(emp_vals):.3f}, "
          f"max={max(emp_vals):.3f}, mean={sum(emp_vals)/len(emp_vals):.3f}")
    print(f"α-CROWN box bound : min={min(box_vals):.3f}, "
          f"max={max(box_vals):.3f}, mean={sum(box_vals)/len(box_vals):.3f}")
    print(f"α-CROWN zono bound: min={min(zono_vals):.3f}, "
          f"max={max(zono_vals):.3f}, mean={sum(zono_vals)/len(zono_vals):.3f}")
    print(f"Mean box/emp ratio:  "
          f"{sum(r[5] for r in results) / len(results):.1f}x")
    print(f"Mean zono/box ratio: "
          f"{sum(r[4]/r[3] for r in results) / len(results)*100:.1f}%")

    # Find smallest 3 (likely most mergeable)
    results_by_bound = sorted(results, key=lambda x: x[3])
    print(f"\n--- Smallest 3 pairs by α-CROWN box (merge candidates) ---")
    for i, j, emp, b_box, b_zono, ratio in results_by_bound[:3]:
        print(f"  ({i},{j}): empirical={emp:.3f}, box={b_box:.3f}, "
              f"zono={b_zono:.3f}")

    # Compare with toy: was 30-300x loose
    box_emp_ratios = [r[3] / max(r[2], 1e-8) for r in results]
    median_ratio = sorted(box_emp_ratios)[len(box_emp_ratios) // 2]
    print(f"\nBound looseness (box/empirical) median: {median_ratio:.1f}x")
    print("  Toy (random init): ~30-300x")
    print("  Pretrained Switch: see above")


if __name__ == "__main__":
    main()
