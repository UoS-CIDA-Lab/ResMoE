"""Experiment 28 — Re-run the bound-driven merge (exp 24) with the routing-
preserving max_logit router, to see if bound ranking now picks good merges.

Exp 24 found a paradox under the DEFAULT router merge (lse_weight): the
empirical output-divergence ORACLE clustering gave the WORST merges and cosine
the BEST, and bound-driven (alpha_crown) collapsed at 8->4. Exp 25 traced the
dominant error to ROUTING change; exp 26 showed max_logit fixes routing
EXACTLY for top-1. So the exp-24 paradox should have been a routing artifact:
once routing is held fixed by max_logit, rel_div reflects ONLY expert-
averaging quality, and the ordering of clusterings may change.

This re-runs exp 24's five clusterings (crown, alpha_crown, cosine,
empirical-oracle, random) on Switch-base-8 top-1, computing rel_div under BOTH
router merges:
  - lse_weight (exp 24's, the framework default)  vs
  - max_logit  (routing-preserving)
Question: with max_logit (routing fixed at ~100%), does the empirical-oracle
stop being worst, and does alpha_crown become competitive with cosine?

Needs HF_HUB_DISABLE_XET=1 (Switch already downloaded).
"""
from __future__ import annotations

import sys
import pathlib
import math
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import (
    crown_diff_bound, alpha_crown_diff_bound, empirical_pair_diff,
)
from cert_moe.baselines import cosine_distance_matrix
from cert_moe.merge import (
    complete_linkage_clusters, MergePlan, _build_label_of,
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


def spearman(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra -= ra.mean(); rb -= rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm()).clamp(min=1e-12)).item()


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def cluster_experts(moe, plan, freq):
    out = []
    for members in plan.clusters:
        fr = freq[members]
        w = fr / fr.sum() if fr.sum() > 0 else torch.ones_like(fr) / len(fr)
        W1 = sum(w[i] * moe.experts[m].W1 for i, m in enumerate(members))
        W2 = sum(w[i] * moe.experts[m].W2 for i, m in enumerate(members))
        out.append((W1, W2))
    return out


@torch.no_grad()
def evaluate(moe, plan, H, freq, mode):
    """(routing_pres, rel_div) for a router-merge mode (top-1 Switch)."""
    logits = H @ moe.router.weight.T
    orig_top1 = logits.argmax(-1)
    label_of = torch.tensor(plan.label_of)
    orig_cluster = label_of[orig_top1]

    if mode == "lse_weight":
        rows = torch.stack([torch.logsumexp(moe.router.weight[m], dim=0)
                            for m in plan.clusters], dim=0)
        cl = H @ rows.T
    elif mode == "max_logit":
        cl = torch.stack([logits[:, m].max(dim=-1).values
                          for m in plan.clusters], dim=-1)
    else:
        raise ValueError(mode)
    sel = cl.argmax(-1)
    routing_pres = (sel == orig_cluster).float().mean().item()

    y_orig = torch.zeros(H.shape[0], moe.cfg.d_model)
    for i in range(moe.cfg.n_experts):
        m = orig_top1 == i
        if m.any():
            e = moe.experts[i]
            y_orig[m] = torch.relu(H[m] @ e.W1.T) @ e.W2.T

    cexp = cluster_experts(moe, plan, freq)
    y_merged = torch.zeros_like(y_orig)
    for c, (W1, W2) in enumerate(cexp):
        m = sel == c
        if m.any():
            y_merged[m] = torch.relu(H[m] @ W1.T) @ W2.T
    rel = (y_orig - y_merged).abs().mean().item() / y_orig.abs().mean().item()
    return routing_pres, rel


def random_plan(n_experts, target_n, gen):
    labels = torch.randint(0, target_n, (n_experts,), generator=gen).tolist()
    for c in range(target_n):
        labels[c] = c
    clusters = [[i for i in range(n_experts) if labels[i] == c]
                for c in range(target_n)]
    clusters = [c for c in clusters if c]
    return MergePlan(clusters=clusters,
                     label_of=_build_label_of(clusters, n_experts))


def main():
    print("Loading Switch-base-8 layer 1...")
    moe, model, tok = load_switch_block()
    N, K = moe.cfg.n_experts, moe.cfg.top_k
    dtype = moe.router.weight.dtype
    H = collect_hidden_states(model, tok, PATH, CALIB, batch_size=8).to(dtype)
    print(f"Calibration: {H.shape[0]} tokens (top_k={K})")

    with torch.no_grad():
        logits = H @ moe.router.weight.T
    top1 = logits.argmax(-1)
    freq = torch.zeros(N)
    freq.scatter_add_(0, top1, torch.ones(H.shape[0]))

    h_lo, h_hi = H.min(0).values - 0.05, H.max(0).values + 0.05

    print("Building distance matrices (crown, alpha_crown, cosine, empirical)...")
    INF = math.inf
    d_crown = torch.full((N, N), INF); d_crown.fill_diagonal_(0)
    d_acrown = torch.full((N, N), INF); d_acrown.fill_diagonal_(0)
    d_emp = torch.full((N, N), INF); d_emp.fill_diagonal_(0)
    pair = []
    for i, j in combinations(range(N), 2):
        ei, ej = moe.experts[i], moe.experts[j]
        bc = crown_diff_bound(ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi)
        ba = alpha_crown_diff_bound(ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi,
                                    n_iters=20)
        be = empirical_pair_diff(ei, ej, H)
        d_crown[i, j] = d_crown[j, i] = bc
        d_acrown[i, j] = d_acrown[j, i] = ba
        d_emp[i, j] = d_emp[j, i] = be
        pair.append((bc, ba, be))
    d_cos = cosine_distance_matrix(moe.experts)

    bc_l = [p[0] for p in pair]; ba_l = [p[1] for p in pair]
    be_l = [p[2] for p in pair]
    cos_l = [d_cos[i, j].item() for i, j in combinations(range(N), 2)]
    print(f"\n  Spearman(crown,       empirical) = {spearman(bc_l, be_l):+.3f}")
    print(f"  Spearman(alpha_crown, empirical) = {spearman(ba_l, be_l):+.3f}")
    print(f"  Spearman(cosine,      empirical) = {spearman(cos_l, be_l):+.3f}")

    methods = {
        "crown": d_crown, "alpha_crown": d_acrown,
        "cosine": d_cos, "empirical(oracle)": d_emp,
    }
    gen = torch.Generator().manual_seed(0)

    for target_n in (6, 4):
        print(f"\n{'='*70}")
        print(f"  Switch 8 -> {target_n}: rel_div under lse_weight vs max_logit")
        print(f"{'='*70}")
        print(f"  {'clustering':18s} {'lse_weight':>22s} {'max_logit':>22s}")
        print(f"  {'':18s} {'rel_div  (route)':>22s} {'rel_div  (route)':>22s}")
        print("  " + "-" * 64)
        for name, dist in methods.items():
            plan = complete_linkage_clusters(dist, find_cutoff(dist, target_n))
            rp_l, rel_l = evaluate(moe, plan, H, freq, "lse_weight")
            rp_m, rel_m = evaluate(moe, plan, H, freq, "max_logit")
            print(f"  {name:18s} {rel_l*100:8.2f}% ({rp_l*100:5.1f}%) "
                  f"{rel_m*100:8.2f}% ({rp_m*100:5.1f}%)")
        # random floor (avg over 20)
        rls, rms = [], []
        for _ in range(20):
            p = random_plan(N, target_n, gen)
            rls.append(evaluate(moe, p, H, freq, "lse_weight")[1])
            rms.append(evaluate(moe, p, H, freq, "max_logit")[1])
        print(f"  {'random (20)':18s} {sum(rls)/len(rls)*100:8.2f}% ( ---- ) "
              f"{sum(rms)/len(rms)*100:8.2f}% ( ---- )")

    print(f"\n{'='*70}")
    print("  VERDICT — with max_logit routing held ~fixed, does the exp-24")
    print("  paradox (oracle worst, cosine best) survive, and is alpha_crown")
    print("  now competitive? Pure expert-averaging quality, routing removed.")


if __name__ == "__main__":
    main()
