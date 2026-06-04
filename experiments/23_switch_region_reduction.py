"""Experiment 23 — Can region reduction make the equivalence bound tighter
AND discriminative?

The documented failure (RESEARCH_OVERVIEW §5.2): on Switch the verified
pairwise bound is ~226x loose and its RANKING is uncorrelated with the
empirical pairwise divergence — so bound-driven merging ≈ random.

Hypothesis: the bound is computed over the FULL calibration box, which is
far wider than the inputs a given expert pair actually receives. If we
verify equivalence only over the routed region (inputs whose top-1 is i
or j), the linear-relaxation slack should shrink and the true signal may
re-emerge — which is also the semantically correct region for deciding a
merge ("are i and j equivalent on the inputs they actually see?").

We compare, for all 28 expert pairs of Switch-base-8 layer 1:
  - full-box     : α-CROWN bound over the whole calibration box
  - routed-box   : α-CROWN bound over the box of inputs routed to i or j
  - routed-zono  : α-CROWN over a PCA-zonotope of the routed inputs
against the empirical worst-case divergence on the matching region.

Metrics: median looseness (bound/empirical) and Spearman rank correlation
between bound and empirical (the discrimination metric).

Needs HF_HUB_DISABLE_XET=1 (Switch already downloaded by exp 21).
"""
from __future__ import annotations

import sys
import pathlib
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import (
    pair_diff_bound, empirical_pair_diff,
    zonotope_from_data, alpha_crown_zonotope_bound,
)

PATH = "encoder.block.1.layer.1.mlp"
CALIB = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "Machine learning models predict outputs from inputs.",
    "Neural networks consist of layers of neurons.",
    "Transformers use attention mechanisms.",
    "Mixture of experts increases model capacity.",
    "Quantum computers may revolutionize computation.",
    "Climate change is a pressing global concern.",
    "Mathematics describes the universe.",
    "Music transcends language barriers.",
    "Education shapes future generations.",
    "Technology drives social change.",
    "Sports build character and camaraderie.",
    "Art expresses what words cannot.",
    "History informs our present and future.",
] * 6
MIN_ROUTED = 8


def spearman(a, b):
    """Rank correlation, manual (no scipy)."""
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra -= ra.mean(); rb -= rb.mean()
    denom = (ra.norm() * rb.norm()).clamp(min=1e-12)
    return (ra @ rb / denom).item()


def box_of(H):
    return H.min(0).values - 0.05, H.max(0).values + 0.05


def main():
    print("Loading Switch-base-8 layer 1...")
    moe, model, tok = load_switch_block()
    K = moe.cfg.top_k
    dtype = moe.router.weight.dtype

    print("Collecting calibration hidden states...")
    H = collect_hidden_states(model, tok, PATH, CALIB, batch_size=8).to(dtype)
    print(f"  {H.shape[0]} tokens")

    with torch.no_grad():
        logits = H @ moe.router.weight.T
    top1 = logits.argmax(-1)  # [T], Switch is top-1

    h_lo_full, h_hi_full = box_of(H)

    rows = []
    print(f"\nComputing bounds for all {moe.cfg.n_experts*(moe.cfg.n_experts-1)//2} pairs...")
    for i, j in combinations(range(moe.cfg.n_experts), 2):
        ei, ej = moe.experts[i], moe.experts[j]
        routed = H[(top1 == i) | (top1 == j)]
        if routed.shape[0] < MIN_ROUTED:
            continue
        # full box
        emp_full = empirical_pair_diff(ei, ej, H)
        b_full = pair_diff_bound(ei, ej, h_lo_full, h_hi_full,
                                 method="alpha_crown", alpha_iters=20)
        # routed box
        h_lo_r, h_hi_r = box_of(routed)
        emp_routed = empirical_pair_diff(ei, ej, routed)
        b_box = pair_diff_bound(ei, ej, h_lo_r, h_hi_r,
                                method="alpha_crown", alpha_iters=20)
        # routed zonotope (PCA)
        k = min(16, routed.shape[0] - 1)
        center, V, acoef = zonotope_from_data(routed, n_components=k, margin=0.05)
        b_zono = alpha_crown_zonotope_bound(ei, ej, center, V, acoef, n_iters=20)
        rows.append({
            "pair": (i, j), "n": routed.shape[0],
            "emp_full": emp_full, "b_full": b_full,
            "emp_routed": emp_routed, "b_box": b_box, "b_zono": b_zono,
        })
        print(f"  ({i},{j}) n={routed.shape[0]:4d} | "
              f"emp_full {emp_full:8.1f} b_full {b_full:9.1f} | "
              f"emp_rt {emp_routed:8.1f} b_box {b_box:9.1f} b_zono {b_zono:9.1f}")

    # Aggregate
    def med(xs):
        xs = sorted(xs)
        return xs[len(xs)//2] if xs else float("nan")

    loose_full = [r["b_full"]/max(r["emp_full"], 1e-6) for r in rows]
    loose_box = [r["b_box"]/max(r["emp_routed"], 1e-6) for r in rows]
    loose_zono = [r["b_zono"]/max(r["emp_routed"], 1e-6) for r in rows]

    print(f"\n{'='*64}")
    print(f"  REGION REDUCTION RESULTS ({len(rows)} pairs)")
    print(f"{'='*64}")
    print(f"  median looseness (bound / empirical):")
    print(f"    full-box   : {med(loose_full):8.1f}x")
    print(f"    routed-box : {med(loose_box):8.1f}x")
    print(f"    routed-zono: {med(loose_zono):8.1f}x")
    print(f"\n  discrimination (Spearman rank corr bound vs empirical;")
    print(f"  1.0 = perfectly ranks pairs by true equivalence, 0 = random):")
    print(f"    full-box    vs emp_full  : "
          f"{spearman([r['b_full'] for r in rows], [r['emp_full'] for r in rows]):+.3f}")
    print(f"    routed-box  vs emp_routed: "
          f"{spearman([r['b_box'] for r in rows], [r['emp_routed'] for r in rows]):+.3f}")
    print(f"    routed-zono vs emp_routed: "
          f"{spearman([r['b_zono'] for r in rows], [r['emp_routed'] for r in rows]):+.3f}")
    print(f"\n  If routed correlation >> full-box correlation, region")
    print(f"  reduction restores the ability to identify equivalent pairs.")


if __name__ == "__main__":
    main()
