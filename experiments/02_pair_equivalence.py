"""Experiment 02 — Validate the certified pair equivalence bound.

Critical question: does our IBP/CROWN bound on ||E_i - E_j||_∞ actually
distinguish planted clone pairs from random pairs?

If the bound is too LOOSE, all pairs look equally "non-mergeable" and
the framework collapses (vacuous). This is the most important sanity
check for the prototype.

We compare:
  - Empirical max ||E_i(h) - E_j(h)||_∞ over calibration samples (lower bound on truth)
  - IBP bound (sound upper bound, loose)
  - CROWN bound (sound upper bound, tighter)

We expect:
  - Clone pairs have low empirical + low bound
  - Random pairs have high empirical + high bound
  - Ranking by bound roughly preserves ranking by truth
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from itertools import combinations

import torch

from cert_moe.toy_moe import ToyMoE, ToyMoEConfig, make_calibration_data
from cert_moe.router_stability import stable_topk_mask
from cert_moe.expert_bounds import (
    pair_diff_bound, empirical_pair_diff
)


def main():
    torch.manual_seed(0)
    cfg = ToyMoEConfig(d_model=64, d_ff=128, n_experts=16, top_k=2,
                       n_clone_pairs=2, clone_noise_std=0.01)
    moe = ToyMoE(cfg)
    H = make_calibration_data(moe, n_samples=512)

    print(f"Planted clone pairs: {moe.clone_pairs}")
    print(f"Clone noise std: {cfg.clone_noise_std}\n")

    # Construct input region from calibration data (axis-aligned bounding box)
    # We use ALL calibration data here for simplicity. The full framework
    # would restrict to stable subset; we cover that in experiment 03.
    h_lo = H.min(0).values
    h_hi = H.max(0).values
    margin = 0.1 * (h_hi - h_lo).mean()  # small generalization margin (ρ)
    h_lo = h_lo - margin
    h_hi = h_hi + margin

    print(f"Input box: per-dim range mean = {(h_hi-h_lo).mean().item():.3f}, "
          f"max = {(h_hi-h_lo).max().item():.3f}")
    print(f"Calibration H norm: mean ||h||_2 = {H.norm(dim=-1).mean().item():.3f}\n")

    # Compute bounds for ALL pairs (small enough to be O(N^2))
    N = cfg.n_experts
    pairs = list(combinations(range(N), 2))

    rows = []
    for i, j in pairs:
        emp = empirical_pair_diff(moe.experts[i], moe.experts[j], H)
        b_ibp = pair_diff_bound(moe.experts[i], moe.experts[j],
                                 h_lo, h_hi, method="ibp")
        b_crown = pair_diff_bound(moe.experts[i], moe.experts[j],
                                   h_lo, h_hi, method="crown")
        is_clone = (i, j) in moe.clone_pairs or (j, i) in moe.clone_pairs
        rows.append((i, j, emp, b_ibp, b_crown, is_clone))

    # Sort by CROWN bound
    rows.sort(key=lambda r: r[4])

    print(f"Pair-wise expert difference bounds (sorted by CROWN bound)")
    print(f"{'pair':>10} | {'empirical':>10} | {'IBP':>10} | {'CROWN':>10} | clone?")
    print("-" * 70)
    # Print top-10 smallest, then a few large for contrast
    for k, (i, j, emp, b_ibp, b_crown, is_clone) in enumerate(rows):
        if k < 10 or is_clone or k >= len(rows) - 5:
            tag = "  ← CLONE" if is_clone else ""
            print(f"  ({i:2d},{j:2d}) | {emp:>10.4f} | {b_ibp:>10.4f} | "
                  f"{b_crown:>10.4f}{tag}")
        elif k == 10:
            print("       ...")

    # Summary statistics
    print()
    clone_rows = [r for r in rows if r[5]]
    non_clone_rows = [r for r in rows if not r[5]]
    print(f"Clone pairs ({len(clone_rows)}):")
    print(f"  emp:    mean={sum(r[2] for r in clone_rows)/len(clone_rows):.4f}, "
          f"max={max(r[2] for r in clone_rows):.4f}")
    print(f"  CROWN:  mean={sum(r[4] for r in clone_rows)/len(clone_rows):.4f}, "
          f"max={max(r[4] for r in clone_rows):.4f}")
    print(f"\nNon-clone pairs ({len(non_clone_rows)}):")
    print(f"  emp:    mean={sum(r[2] for r in non_clone_rows)/len(non_clone_rows):.4f}, "
          f"min={min(r[2] for r in non_clone_rows):.4f}")
    print(f"  CROWN:  mean={sum(r[4] for r in non_clone_rows)/len(non_clone_rows):.4f}, "
          f"min={min(r[4] for r in non_clone_rows):.4f}")

    # Bound tightness: how loose is CROWN vs empirical?
    print()
    print("Bound tightness (CROWN / empirical ratio):")
    print(f"  clone pairs: mean ratio = "
          f"{sum(r[4]/max(r[2],1e-6) for r in clone_rows)/len(clone_rows):.2f}x")
    print(f"  non-clone:   mean ratio = "
          f"{sum(r[4]/max(r[2],1e-6) for r in non_clone_rows)/len(non_clone_rows):.2f}x")

    # Ranking quality: how well does CROWN rank match empirical rank?
    crown_rank = sorted(range(len(rows)), key=lambda k: rows[k][4])
    emp_rank = sorted(range(len(rows)), key=lambda k: rows[k][2])
    # Spearman-like: agreement on top-k mergeable candidates
    for top_k in (4, 8, 16):
        crown_top = set(crown_rank[:top_k])
        emp_top = set(emp_rank[:top_k])
        overlap = len(crown_top & emp_top) / top_k
        print(f"  top-{top_k} overlap (CROWN ranking vs empirical): {overlap*100:.0f}%")


if __name__ == "__main__":
    main()
