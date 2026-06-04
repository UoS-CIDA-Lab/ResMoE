"""Experiment 29 — Why does bound-driven merging collapse at 8->4?

Exp 28 (max_logit, routing fixed at 100%) showed CROWN beats cosine at the
mild ratio 8->6 (10.43% vs 26.80%) but collapses to ~random at 8->4 (55.72%
vs cosine 37.00%). With routing held fixed by max_logit, rel_div is PURE
expert-averaging error, so we can dissect the collapse cleanly. Two questions:

  Q1 RANKING vs CUTOFF: is crown's 8->4 collapse because its pairwise RANKING
     stops matching what averages well, or because complete-linkage/cutoff is
     forced into a bad partition? Switch has N=8, so we BRUTE-FORCE the truly
     optimal partition (min rel_div) for k=6 and k=4 (Stirling2 = 266 / 1701)
     and ask: how far are crown/cosine from optimal, and do the OPTIMAL merged
     pairs have low crown bound or low cosine distance?

  Q2 LOCALIZATION: attribute rel_div to individual clusters (each token routes
     to exactly one cluster under max_logit), to see if one forced
     non-equivalent merge causes the 8->4 blowup, for crown vs cosine.

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


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def set_partitions(elements, k):
    """Yield all partitions of `elements` into exactly k non-empty blocks."""
    n = len(elements)
    if k < 1 or k > n:
        return
    def helper(i, blocks):
        if i == n:
            if len(blocks) == k:
                yield [list(b) for b in blocks]
            return
        # too few remaining to fill k blocks?
        if len(blocks) + (n - i) < k:
            return
        e = elements[i]
        for b in range(len(blocks)):
            blocks[b].append(e)
            yield from helper(i + 1, blocks)
            blocks[b].pop()
        if len(blocks) < k:
            blocks.append([e])
            yield from helper(i + 1, blocks)
            blocks.pop()
    yield from helper(0, [])


def plan_of(clusters, N):
    return MergePlan(clusters=clusters, label_of=_build_label_of(clusters, N))


@torch.no_grad()
def attrib(moe, plan, H, freq):
    """max_logit routing (top-1). Returns (rel_div, per-cluster contrib list).

    per-cluster: (members, n_tokens, err_contrib%, max_crown, max_cos) filled
    by caller for the distance fields.
    """
    logits = H @ moe.router.weight.T
    orig_top1 = logits.argmax(-1)
    label_of = torch.tensor(plan.label_of)
    cl = torch.stack([logits[:, m].max(dim=-1).values
                      for m in plan.clusters], dim=-1)
    sel = cl.argmax(-1)

    y_orig = torch.zeros(H.shape[0], moe.cfg.d_model)
    for i in range(moe.cfg.n_experts):
        m = orig_top1 == i
        if m.any():
            e = moe.experts[i]
            y_orig[m] = torch.relu(H[m] @ e.W1.T) @ e.W2.T

    # cluster experts (freq-weighted avg)
    cexp = []
    for members in plan.clusters:
        fr = freq[members]
        w = fr / fr.sum() if fr.sum() > 0 else torch.ones_like(fr) / len(fr)
        W1 = sum(w[i] * moe.experts[m].W1 for i, m in enumerate(members))
        W2 = sum(w[i] * moe.experts[m].W2 for i, m in enumerate(members))
        cexp.append((W1, W2))

    y_merged = torch.zeros_like(y_orig)
    for c, (W1, W2) in enumerate(cexp):
        m = sel == c
        if m.any():
            y_merged[m] = torch.relu(H[m] @ W1.T) @ W2.T

    err = (y_orig - y_merged).abs().sum(-1)        # [B] per-token L1
    denom = y_orig.abs().sum()
    rel = err.sum().item() / denom.item()

    contrib = []
    for c, members in enumerate(plan.clusters):
        m = sel == c
        contrib.append({
            "members": members,
            "n_tokens": int(m.sum().item()),
            "err_pct": err[m].sum().item() / denom.item() * 100,
        })
    return rel, contrib


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
    d_crown = torch.full((N, N), math.inf); d_crown.fill_diagonal_(0)
    d_acrown = torch.full((N, N), math.inf); d_acrown.fill_diagonal_(0)
    d_emp = torch.full((N, N), math.inf); d_emp.fill_diagonal_(0)
    for i, j in combinations(range(N), 2):
        ei, ej = moe.experts[i], moe.experts[j]
        d_crown[i, j] = d_crown[j, i] = crown_diff_bound(
            ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi)
        d_acrown[i, j] = d_acrown[j, i] = alpha_crown_diff_bound(
            ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi, n_iters=20)
        d_emp[i, j] = d_emp[j, i] = empirical_pair_diff(ei, ej, H)
    d_cos = cosine_distance_matrix(moe.experts)

    def merged_pair_stats(clusters):
        """mean crown / cosine / empirical over within-cluster pairs."""
        cr, co, em, npair = 0.0, 0.0, 0.0, 0
        for members in clusters:
            for i, j in combinations(members, 2):
                cr += d_crown[i, j].item(); co += d_cos[i, j].item()
                em += d_emp[i, j].item(); npair += 1
        if npair == 0:
            return None
        return cr / npair, co / npair, em / npair, npair

    methods = {"crown": d_crown, "alpha_crown": d_acrown,
               "cosine": d_cos, "empirical(oracle)": d_emp}

    for k in (6, 4):
        print(f"\n{'='*72}")
        print(f"  Switch 8 -> {k}   (max_logit routing, PURE averaging error)")
        print(f"{'='*72}")

        # named clusterings
        named = {}
        for name, dist in methods.items():
            plan = complete_linkage_clusters(dist, find_cutoff(dist, k))
            named[name] = plan

        # brute-force optimum
        best_rel, best_plan = math.inf, None
        count = 0
        for blocks in set_partitions(list(range(N)), k):
            count += 1
            rel, _ = attrib(moe, plan_of(blocks, N), H, freq)
            if rel < best_rel:
                best_rel, best_plan = rel, [list(b) for b in blocks]
        named["OPTIMAL"] = plan_of(best_plan, N)
        print(f"  brute-forced {count} partitions; optimum rel_div "
              f"{best_rel*100:.2f}%\n")

        print(f"  {'method':18s} {'rel_div':>8s}  {'merged-pair means':>30s}")
        print(f"  {'':18s} {'':>8s}  {'crown':>9s} {'cosine':>8s} {'emp':>8s}")
        print("  " + "-" * 60)
        for name, plan in named.items():
            rel, _ = attrib(moe, plan, H, freq)
            st = merged_pair_stats(plan.clusters)
            if st:
                cr, co, em, _ = st
                print(f"  {name:18s} {rel*100:7.2f}%  {cr:9.3f} {co:8.4f} "
                      f"{em:8.4f}")
            else:
                print(f"  {name:18s} {rel*100:7.2f}%  (no merges)")

        # localization: clusters + per-cluster error for crown vs cosine vs OPT
        for name in ("crown", "cosine", "OPTIMAL"):
            plan = named[name]
            rel, contrib = attrib(moe, plan, H, freq)
            contrib.sort(key=lambda d: -d["err_pct"])
            print(f"\n  [{name}] clusters (sorted by error contribution):")
            for c in contrib:
                tag = "" if len(c["members"]) > 1 else "  (singleton)"
                print(f"    {str(c['members']):20s} "
                      f"n={c['n_tokens']:4d}  err={c['err_pct']:5.1f}%{tag}")

    print(f"\n{'='*72}")
    print("  READ: if OPTIMAL's merged-pair COSINE mean is low but CROWN mean")
    print("  is high at 8->4, crown's collapse is a RANKING failure (the true")
    print("  best merges are cosine-close, not crown-close). If crown≈OPTIMAL,")
    print("  the collapse is a cutoff/linkage artifact, not ranking.")


if __name__ == "__main__":
    main()
