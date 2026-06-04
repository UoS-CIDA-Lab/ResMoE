"""Experiment 30 — Frequency-aware bound-driven merge fixes the 8->4 collapse?

Exp 29 traced bound-driven's 8->4 collapse to FREQUENCY-BLINDNESS: complete-
linkage on the crown matrix lumped the dominant expert (43% of tokens) into a
giant cluster. The optimum (and cosine) protect high-traffic experts. Neither
crown nor cosine encodes routing frequency in its distance.

Two fixes, applied to cosine / crown / alpha_crown, vs the brute-forced
OPTIMAL (Switch N=8, top-1, max_logit routing -> pure averaging error):
  A) freq-weighted distance  d'[i,j] = d[i,j] * (freq[i]+freq[j])
     (merge total error ~ tokens_affected * pairwise_divergence, so this is
      the principled distance to minimize total error)
  B) protect-top-T: force the T highest-traffic experts to stay singletons,
     cluster the rest into (k - T) clusters with the base distance.

Question: does a freq-aware criterion let the near-optimal-at-8->6 bound
ranking extend to 8->4, closing the gap to OPTIMAL (15.65%)?

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
from cert_moe.expert_bounds import crown_diff_bound, alpha_crown_diff_bound
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


def plan_of(clusters, N):
    return MergePlan(clusters=clusters, label_of=_build_label_of(clusters, N))


def set_partitions(elements, k):
    n = len(elements)
    if k < 1 or k > n:
        return
    def helper(i, blocks):
        if i == n:
            if len(blocks) == k:
                yield [list(b) for b in blocks]
            return
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


def cluster_plain(dist, k, N):
    return complete_linkage_clusters(dist, find_cutoff(dist, k))


def cluster_protect(dist, freq, k, T, N):
    prot = freq.argsort(descending=True)[:T].tolist()
    rest = [i for i in range(N) if i not in prot]
    if k - T < 1 or len(rest) < k - T:
        return None
    sub = dist[rest][:, rest].clone()
    subplan = complete_linkage_clusters(sub, find_cutoff(sub, k - T))
    clusters = [[rest[m] for m in members] for members in subplan.clusters]
    clusters += [[p] for p in prot]
    return plan_of(clusters, N)


def freq_weight(dist, freq):
    d = dist * (freq.unsqueeze(0) + freq.unsqueeze(1))
    d.fill_diagonal_(0.0)
    return d


@torch.no_grad()
def rel_div_maxlogit(moe, plan, H, freq):
    logits = H @ moe.router.weight.T
    orig_top1 = logits.argmax(-1)
    cl = torch.stack([logits[:, m].max(dim=-1).values
                      for m in plan.clusters], dim=-1)
    sel = cl.argmax(-1)

    y_orig = torch.zeros(H.shape[0], moe.cfg.d_model)
    for i in range(moe.cfg.n_experts):
        m = orig_top1 == i
        if m.any():
            e = moe.experts[i]
            y_orig[m] = torch.relu(H[m] @ e.W1.T) @ e.W2.T

    y_merged = torch.zeros_like(y_orig)
    for c, members in enumerate(plan.clusters):
        fr = freq[members]
        w = fr / fr.sum() if fr.sum() > 0 else torch.ones_like(fr) / len(fr)
        W1 = sum(w[i] * moe.experts[m].W1 for i, m in enumerate(members))
        W2 = sum(w[i] * moe.experts[m].W2 for i, m in enumerate(members))
        m = sel == c
        if m.any():
            y_merged[m] = torch.relu(H[m] @ W1.T) @ W2.T
    return (y_orig - y_merged).abs().sum().item() / y_orig.abs().sum().item()


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
    print(f"Routing freq: {freq.int().tolist()} "
          f"(dominant expert {int(freq.argmax())} = {int(freq.max())} tokens)")

    h_lo, h_hi = H.min(0).values - 0.05, H.max(0).values + 0.05
    d_crown = torch.full((N, N), math.inf); d_crown.fill_diagonal_(0)
    d_acrown = torch.full((N, N), math.inf); d_acrown.fill_diagonal_(0)
    for i, j in combinations(range(N), 2):
        ei, ej = moe.experts[i], moe.experts[j]
        d_crown[i, j] = d_crown[j, i] = crown_diff_bound(
            ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi)
        d_acrown[i, j] = d_acrown[j, i] = alpha_crown_diff_bound(
            ei.W1, ei.W2, ej.W1, ej.W2, h_lo, h_hi, n_iters=20)
    d_cos = cosine_distance_matrix(moe.experts)

    base = {"cosine": d_cos, "crown": d_crown, "alpha_crown": d_acrown}
    fw = {f"{n}+freqw": freq_weight(d, freq) for n, d in base.items()}

    for k in (6, 4):
        print(f"\n{'='*68}")
        print(f"  Switch 8 -> {k}  (rel_div, max_logit; lower=better)")
        print(f"{'='*68}")
        # brute-force optimum
        best = math.inf
        for blocks in set_partitions(list(range(N)), k):
            best = min(best, rel_div_maxlogit(moe, plan_of(blocks, N), H, freq))
        print(f"  OPTIMAL (brute force)        {best*100:6.2f}%")
        print("  " + "-" * 40)
        # plain
        for n, d in base.items():
            r = rel_div_maxlogit(moe, cluster_plain(d, k, N), H, freq)
            print(f"  {n:24s}     {r*100:6.2f}%")
        print("  " + "-" * 40 + "  [A] freq-weighted distance")
        for n, d in fw.items():
            r = rel_div_maxlogit(moe, cluster_plain(d, k, N), H, freq)
            print(f"  {n:24s}     {r*100:6.2f}%")
        print("  " + "-" * 40 + "  [B] protect top-T traffic")
        for n, d in base.items():
            for T in (1, 2):
                p = cluster_protect(d, freq, k, T, N)
                if p is not None:
                    r = rel_div_maxlogit(moe, p, H, freq)
                    print(f"  {n+f'+protect{T}':24s}     {r*100:6.2f}%")

    print(f"\n{'='*68}")
    print("  READ: if crown+freqw / crown+protect approach OPTIMAL at 8->4")
    print("  (vs plain crown 55.7%), frequency-awareness is the missing piece")
    print("  and bound-driven merging is rehabilitated at aggressive ratios.")


if __name__ == "__main__":
    main()
