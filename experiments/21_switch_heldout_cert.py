"""Experiment 21 — Does Switch-base-8 cert% survive on held-out tokens?

Experiment 20 found that on OLMoE the cert-aware cert% (28.6% on the 70
calibration tokens) collapsed to 0.1% on real eval tokens — the merge plan
overfits the calibration set. RESEARCH_OVERVIEW reports Switch-base-8 8→4
cert-aware cert% = 88%, but that number (exp 09) was measured on the SAME
hidden states used to build the plan. This re-validates it:

  - build the cert-aware plan (8→4) on a CALIBRATION token set
  - measure cert% on calibration AND on a disjoint HELD-OUT token set
    (different sentences, same domain)

If held-out cert% stays near 88%, the claim is robust and the OLMoE
collapse is specific to its harder 64-expert/top-8 regime. If held-out
cert% collapses, the core paper claim is in trouble across the board.

Needs HF_HUB_DISABLE_XET=1 to download google/switch-base-8.
Run: HF_HUB_DISABLE_XET=1 python3 experiments/21_switch_heldout_cert.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.merge import complete_linkage_clusters, cert_aware_greedy_merge
from cert_moe.theorem1_conditions import check_conditions
from cert_moe.baselines import cosine_distance_matrix

PATH = "encoder.block.1.layer.1.mlp"

# Calibration text — same style as exp 09 (which reported 88%).
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

# Held-out text — DIFFERENT sentences, same general (English prose) domain.
HELDOUT = [
    "The ancient library held thousands of forgotten manuscripts.",
    "Rivers carve canyons over millions of years.",
    "She planted tomatoes and basil in the spring garden.",
    "Economic policy affects employment and inflation alike.",
    "The orchestra tuned their instruments before the concert.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Travelers crossed the desert under a blazing sun.",
    "The committee debated the proposal for several hours.",
    "Bridges must withstand both wind and heavy traffic.",
    "Children learn language with remarkable speed.",
    "The telescope revealed distant galaxies and nebulae.",
    "Fermentation has been used to preserve food for centuries.",
    "Volcanic eruptions reshape the surrounding landscape.",
    "The lawyer presented evidence to the skeptical jury.",
    "Migratory birds navigate using the earth's magnetic field.",
    "A good recipe balances salt, acid, fat, and heat.",
    "The startup raised funding to expand its operations.",
    "Glaciers store a vast amount of the world's fresh water.",
    "Poets often find meaning in ordinary moments.",
    "The surgeon explained the procedure to the nervous patient.",
] * 5


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def cert_breakdown(H, router_w, plan, K):
    m = check_conditions(H, router_w, plan, K=K, delta_threshold=0.1)
    return {
        "mc": m.mc.float().mean().item(),
        "ts": m.ts.float().mean().item(),
        "em": m.em.float().mean().item(),
        "cert": m.all_certified.float().mean().item(),
    }


def main():
    print("Loading Switch-base-8 first MoE block...")
    moe, model, tok = load_switch_block()
    K = moe.cfg.top_k
    dtype = moe.router.weight.dtype

    print("Collecting CALIBRATION hidden states...")
    H_cal = collect_hidden_states(model, tok, PATH, CALIB, batch_size=8).to(dtype)
    print("Collecting HELD-OUT hidden states...")
    H_held = collect_hidden_states(model, tok, PATH, HELDOUT, batch_size=8).to(dtype)
    print(f"  calibration tokens: {H_cal.shape[0]}, "
          f"held-out tokens: {H_held.shape[0]}")

    router_w = moe.router.weight.detach()
    d_cos = cosine_distance_matrix(moe.experts)
    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001

    TARGET_N = 4
    print(f"\n{'='*70}")
    print(f"  Switch-base-8 encoder block 1, 8 → {TARGET_N}")
    print(f"{'='*70}")

    plans = {}
    # cosine (complete-linkage)
    eps = find_cutoff(d_cos, TARGET_N)
    plans["cosine"] = complete_linkage_clusters(d_cos, epsilon=eps)
    # cert-aware built ON CALIBRATION (bound_weight=0, as exp 09)
    plans["cert-aware"] = cert_aware_greedy_merge(
        moe, H_cal, d_cos, target_n_clusters=TARGET_N,
        max_bound=max_b, delta_em=0.1, bound_weight=0.0,
    )

    print(f"\n  {'method':12s} {'set':12s} "
          f"{'MC%':>6s} {'TS%':>6s} {'EM%':>6s} {'cert%':>7s}")
    print("  " + "-" * 52)
    summary = {}
    for name, plan in plans.items():
        for set_name, H in (("calibration", H_cal), ("held-out", H_held)):
            r = cert_breakdown(H, router_w, plan, K)
            summary[(name, set_name)] = r["cert"]
            print(f"  {name:12s} {set_name:12s} "
                  f"{r['mc']*100:>5.1f}% {r['ts']*100:>5.1f}% "
                  f"{r['em']*100:>5.1f}% {r['cert']*100:>6.1f}%")

    print(f"\n{'='*70}")
    print("  VERDICT — cert% calibration → held-out")
    print(f"{'='*70}")
    for name in plans:
        c = summary[(name, "calibration")] * 100
        h = summary[(name, "held-out")] * 100
        drop = "—" if c == 0 else f"{(c-h)/c*100:+.0f}% rel."
        print(f"  {name:12s}: {c:5.1f}%  →  {h:5.1f}%   ({drop})")
    print("\n  If cert-aware held-out ≈ calibration → claim robust.")
    print("  If it collapses (like OLMoE 28.6%→0.1%) → claim overfit.")


if __name__ == "__main__":
    main()
