"""Experiment 10 — Adversarial PGD on Switch-base-8.

The make-or-break test: does our toy claim (cert-region PGD reduces
worst-case divergence by ~50%) replicate on a pretrained model?

Hypothesis (after Stage 3 finding that K=1 changes the game):
  - Free PGD on cert-aware merged MoE: similar or slightly worse than
    cosine (because cert-aware sacrifices raw divergence for cert%)
  - Cert-region projected PGD on cert-aware: significantly bounded
    because cert region is large (87.6% cert% at 8→4) and TS/EM
    conditions actively block adversarial directions

If hypothesis holds → main paper claim replicates on real model.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
from itertools import combinations

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import pair_diff_bound
from cert_moe.merge import (
    complete_linkage_clusters, apply_merge, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions
from cert_moe.baselines import cosine_distance_matrix


SAMPLES = [
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
        plan = complete_linkage_clusters(dist, epsilon=eps)
        if len(plan.clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def free_pgd_attack(moe, merged, h_init, n_steps, step_size, lo, hi):
    """Max ||M(h)-M̂(h)||_inf, project onto bounding box."""
    h = h_init.clone().detach().requires_grad_(True)
    best_div = -1.0
    for _ in range(n_steps):
        y_o, _ = moe(h)
        y_m, _ = merged(h)
        diff = (y_o - y_m).abs()
        loss = -diff.max(-1).values.sum()
        loss.backward()
        with torch.no_grad():
            h += step_size * h.grad.sign()
            h.clamp_(lo, hi)
            h.grad = None
            y_o, _ = moe(h)
            y_m, _ = merged(h)
            cur = (y_o - y_m).abs().max(-1).values.max().item()
            best_div = max(best_div, cur)
        h.requires_grad_(True)
    return best_div


def cert_pgd_attack(moe, merged, plan, h_init, K,
                    n_steps, step_size, lo, hi, delta_em=0.1):
    """PGD that backs off when leaving Theorem 1 certified region."""
    W_g = moe.router.weight.detach()
    masks0 = check_conditions(h_init, W_g, plan, K=K, delta_threshold=delta_em)
    if not masks0.all_certified.all():
        return -1.0, 0

    h = h_init.clone().detach()
    best_div = -1.0
    cert_steps = 0
    for _ in range(n_steps):
        h_g = h.clone().requires_grad_(True)
        y_o, _ = moe(h_g)
        y_m, _ = merged(h_g)
        diff = (y_o - y_m).abs()
        loss = -diff.max(-1).values.sum()
        loss.backward()
        grad_sign = h_g.grad.sign().detach()

        accepted = False
        for s in (step_size, step_size / 2, step_size / 4, step_size / 8):
            with torch.no_grad():
                h_trial = (h + s * grad_sign).clamp(lo, hi)
                masks = check_conditions(h_trial, W_g, plan, K=K,
                                          delta_threshold=delta_em)
                if masks.all_certified.all():
                    h = h_trial
                    cert_steps += 1
                    accepted = True
                    break
        if not accepted:
            break

        with torch.no_grad():
            y_o, _ = moe(h)
            y_m, _ = merged(h)
            cur = (y_o - y_m).abs().max(-1).values.max().item()
            best_div = max(best_div, cur)
    return best_div, cert_steps


def main():
    print("Loading Switch-base-8 MoE block...")
    moe, model, tok = load_switch_block()

    print("Collecting calibration hidden states...")
    H = collect_hidden_states(
        model, tok, "encoder.block.1.layer.1.mlp", SAMPLES, batch_size=8,
    ).to(moe.router.weight.dtype)
    if H.shape[0] > 1024:
        H = H[torch.randperm(H.shape[0])[:1024]]
    print(f"Calibration: {H.shape}")

    # Frequency
    with torch.no_grad():
        logits = H @ moe.router.weight.T
    _, topk_idx = logits.topk(moe.cfg.top_k, dim=-1)
    freq = torch.zeros(moe.cfg.n_experts)
    for k in range(moe.cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))

    # Attack box: per-dim min/max with margin
    lo = H.min(0).values - 0.5
    hi = H.max(0).values + 0.5
    box_diag = (hi - lo).norm().item()
    print(f"Attack box diag: {box_diag:.1f}, ||h||_2 mean: "
          f"{H.norm(dim=-1).mean().item():.1f}")

    # PGD step size: scale relative to per-dim spread
    step_size = (hi - lo).mean().item() * 0.02  # ~2% of typical dim range
    print(f"PGD step size: {step_size:.3f}")

    # Build distance matrices (cosine + bound for cert-aware)
    print("\nBuilding distance matrices...")
    d_cos = cosine_distance_matrix(moe.experts)
    print("  cosine done")

    # Quick CROWN-only bound (no α optimization) for speed
    N = moe.cfg.n_experts
    d_bound = torch.full((N, N), math.inf)
    d_bound.fill_diagonal_(0.0)
    h_box_lo = H.min(0).values - 0.05
    h_box_hi = H.max(0).values + 0.05
    for i, j in combinations(range(N), 2):
        b = pair_diff_bound(moe.experts[i], moe.experts[j],
                            h_box_lo, h_box_hi, method="crown")
        d_bound[i, j] = b
        d_bound[j, i] = b
    print("  bound done")

    target_n = 4   # 8 → 4 (heaviest from Stage 3)
    print(f"\nCompression: 8 → {target_n}")

    # Cosine merge plan
    eps_cos = find_cutoff(d_cos, target_n)
    plan_cos = complete_linkage_clusters(d_cos, epsilon=eps_cos)
    merged_cos = apply_merge(moe, plan_cos, freq)
    print(f"  cosine plan: {plan_cos.clusters}")

    # Bound merge plan
    eps_b = find_cutoff(d_bound, target_n)
    plan_b = complete_linkage_clusters(d_bound, epsilon=eps_b)
    merged_b = apply_merge(moe, plan_b, freq)
    print(f"  bound  plan: {plan_b.clusters}")

    # Cert-aware plan
    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001
    plan_ca = cert_aware_greedy_merge(
        moe, H, d_cos,
        target_n_clusters=target_n,
        max_bound=max_b, delta_em=0.1, bound_weight=0.0,
    )
    merged_ca = apply_merge(moe, plan_ca, freq)
    print(f"  cert-aware plan: {plan_ca.clusters}")

    # Attack setup
    n_starts = 10
    n_steps = 80

    # Pick random starts from within bounding box
    torch.manual_seed(0)
    starts = (lo + (hi - lo)
              * torch.rand(n_starts, moe.cfg.d_model)).unsqueeze(1)
    # starts: [n_starts, 1, d_model]

    print(f"\n{'='*88}")
    print(f"Adversarial PGD ({n_starts} starts × {n_steps} steps)")
    print(f"{'='*88}")

    # Free PGD on each merged model
    print(f"\n{'method':18s} | {'avg div (calib)':>15s} | "
          f"{'worst free-PGD':>15s}")
    print("-" * 65)
    for name, merged in [("cosine", merged_cos),
                          ("bound", merged_b),
                          ("cert-aware", merged_ca)]:
        # Avg divergence (on calibration, no attack)
        with torch.no_grad():
            y_o, _ = moe(H)
            y_m, _ = merged(H)
            avg_div = (y_o - y_m).abs().max(-1).values.mean().item()

        worst = -1.0
        for s in range(n_starts):
            h0 = starts[s]
            d = free_pgd_attack(moe, merged, h0,
                                n_steps=n_steps, step_size=step_size,
                                lo=lo, hi=hi)
            worst = max(worst, d)
        print(f"  {name:18s} | {avg_div:>15.3f} | {worst:>15.3f}")

    # Cert-region PGD (cert-aware only — has the largest cert region)
    print(f"\n{'='*88}")
    print("Cert-region projected PGD (cert-aware only)")
    print(f"{'='*88}")
    masks = check_conditions(H, moe.router.weight.detach(),
                              plan_ca, K=moe.cfg.top_k, delta_threshold=0.1)
    cert_pool = H[masks.all_certified]
    n_cert = cert_pool.shape[0]
    print(f"\nCertified pool: {n_cert}/{H.shape[0]} = "
          f"{n_cert/H.shape[0]*100:.1f}%")
    if n_cert < 5:
        print("  Not enough certified seeds — aborting.")
        return

    n_cert_starts = min(n_starts, n_cert)
    worst_cert_pgd = -1.0
    base_max = 0.0
    total_steps = 0
    for s in range(n_cert_starts):
        h0 = cert_pool[s:s + 1]
        with torch.no_grad():
            y_o, _ = moe(h0)
            y_m, _ = merged_ca(h0)
            base = (y_o - y_m).abs().max(-1).values.max().item()
        base_max = max(base_max, base)

        d, steps = cert_pgd_attack(
            moe, merged_ca, plan_ca, h0, K=moe.cfg.top_k,
            n_steps=n_steps, step_size=step_size,
            lo=lo, hi=hi, delta_em=0.1,
        )
        worst_cert_pgd = max(worst_cert_pgd, d)
        total_steps += steps

    avg_steps = total_steps / n_cert_starts
    print(f"\n  Baseline div (no attack) : {base_max:.3f}")
    print(f"  Cert-PGD worst           : {worst_cert_pgd:.3f}")
    print(f"  Avg cert steps           : {avg_steps:.1f} / {n_steps}")

    print(f"\n{'='*88}")
    print("Summary — main paper claim on Switch-base-8 8→4:")
    print(f"{'='*88}")
    print("Free PGD on heuristic-merged:    worst-case = (cosine row above)")
    print("Free PGD on cert-aware-merged:   worst-case = (cert-aware row)")
    print(f"Cert-PGD on cert-aware:          worst-case = {worst_cert_pgd:.3f}")
    print("If cert-PGD < free-PGD/2 → main claim holds on real model.")
    print(f"{'='*88}")


if __name__ == "__main__":
    main()
