"""Experiment 26 — Is the routing-change error fixable by a better router merge?

Exp 25 showed the dominant merge error is ROUTING change, not expert
approximation: combining router rows with log-sum-exp reroutes tokens to the
wrong cluster. For top-1 routing there is a router merge that preserves
routing EXACTLY: give each cluster the MAX of its members' logits. Then the
cluster containing the original argmax expert has the max cluster score, so
the top-1 decision is preserved. log-sum-exp instead inflates each cluster by
~log|C| and breaks it.

If true, then COSINE clustering (good expert averaging) + MAX-pool router
(routing preserved) should give BOTH low rel_div AND high routing
preservation — resolving the routing↔expert tension that has dogged every
experiment today.

We compare, for fixed clusterings (cosine, cert-aware), three router-merge
schemes, with freq-weighted averaged experts, on Switch-base-8 (top-1):
  - lse_weight : log-sum-exp on router WEIGHT rows (what apply_merge does)
  - lse_logit  : log-sum-exp on LOGITS (what the TS check uses)
  - max_logit  : max of member logits per cluster (routing-preserving)

Metrics: routing-preservation% (merged top-1 cluster == cluster of original
top-1 expert) and rel_div at the MoE layer.

Needs HF_HUB_DISABLE_XET=1 (Switch already downloaded).
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.baselines import cosine_distance_matrix
from cert_moe.merge import complete_linkage_clusters, cert_aware_greedy_merge

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


def cluster_experts(moe, plan, freq):
    """Freq-weighted averaged (W1, W2) per cluster."""
    out = []
    for members in plan.clusters:
        fr = torch.tensor([freq[m].item() for m in members])
        w = fr / fr.sum() if fr.sum() > 0 else torch.ones_like(fr) / len(fr)
        W1 = sum(w[i] * moe.experts[m].W1 for i, m in enumerate(members))
        W2 = sum(w[i] * moe.experts[m].W2 for i, m in enumerate(members))
        out.append((W1, W2))
    return out


def cluster_logits(logits, plan, mode):
    cols = []
    for members in plan.clusters:
        sub = logits[:, members]
        if mode == "lse_logit":
            cols.append(torch.logsumexp(sub, dim=-1))
        elif mode == "max_logit":
            cols.append(sub.max(dim=-1).values)
        else:
            raise ValueError(mode)
    return torch.stack(cols, dim=-1)


def lse_weight_router(moe, plan):
    rows = [torch.logsumexp(moe.router.weight[m], dim=0) for m in plan.clusters]
    return torch.stack(rows, dim=0)  # [n_clusters, d]


@torch.no_grad()
def evaluate(moe, plan, H, freq, mode):
    """Return (routing_preservation, rel_div) for a router-merge mode."""
    logits = H @ moe.router.weight.T          # [B, N]
    orig_top1 = logits.argmax(-1)             # [B]
    label_of = torch.tensor(plan.label_of)
    orig_cluster = label_of[orig_top1]        # [B]

    if mode == "lse_weight":
        Wc = lse_weight_router(moe, plan)
        cl = H @ Wc.T
    else:
        cl = cluster_logits(logits, plan, mode)
    sel = cl.argmax(-1)                        # [B] merged top-1 cluster
    routing_pres = (sel == orig_cluster).float().mean().item()

    # original output (top-1, gate=1): per-token original expert output
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


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


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

    d_cos = cosine_distance_matrix(moe.experts)

    for target_n in (6, 4):
        print(f"\n{'='*68}")
        print(f"  Switch 8 → {target_n}: routing-preservation% | rel_div")
        print(f"{'='*68}")
        clusterings = {
            "cosine": complete_linkage_clusters(d_cos, find_cutoff(d_cos, target_n)),
            "cert-aware": cert_aware_greedy_merge(
                moe, H, d_cos, target_n_clusters=target_n,
                max_bound=d_cos[d_cos.isfinite()].max().item()*1.001,
                delta_em=0.1, bound_weight=0.0),
        }
        print(f"  {'clustering':12s} {'router':12s} {'route-pres':>11s} {'rel_div':>9s}")
        print("  " + "-" * 50)
        for cname, plan in clusterings.items():
            for mode in ("lse_weight", "lse_logit", "max_logit"):
                rp, rel = evaluate(moe, plan, H, freq, mode)
                print(f"  {cname:12s} {mode:12s} {rp*100:>10.1f}% {rel*100:>8.2f}%")

    print(f"\n{'='*68}")
    print("  KEY: does cosine + max_logit give high route-pres AND low rel_div?")
    print("  If yes → routing was a fixable router-merge artifact, and the")
    print("  expert↔routing tension dissolves.")


if __name__ == "__main__":
    main()
