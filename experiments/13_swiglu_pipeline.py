"""Experiment 13 — Full Stage 1-4 pipeline on synthetic SwiGLU MoE.

Goal: verify the SwiGLU bound (cert_moe.swiglu_bounds) integrates
cleanly with the rest of the framework, end-to-end. Compare with the
ReLU toy results to see how activation choice affects each stage.

Pipeline:
  Stage 1: router stability sweep
  Stage 2: pair-wise SwiGLU bound (compared to empirical)
  Stage 3: cosine vs cert-aware merging
  Stage 4: adversarial PGD (free + cert-region projected)
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
from itertools import combinations

import torch

from cert_moe.toy_swiglu_moe import (
    ToySwiGLUMoE, ToySwiGLUMoEConfig, make_calibration_data,
    apply_merge_swiglu, SwiGLUExpert,
)
from cert_moe.router_stability import stable_topk_mask
from cert_moe.swiglu_bounds import swiglu_diff_bound_box
from cert_moe.merge import (
    complete_linkage_clusters, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions
from cert_moe.baselines import cosine_distance_matrix


def empirical_swiglu_diff(expert_i, expert_j, H):
    with torch.no_grad():
        d = expert_i(H) - expert_j(H)
    return d.abs().max().item()


def swiglu_bound_matrix(moe, h_lo, h_hi):
    N = moe.cfg.n_experts
    delta = torch.full((N, N), math.inf)
    delta.fill_diagonal_(0.0)
    for i, j in combinations(range(N), 2):
        b = swiglu_diff_bound_box(
            moe.experts[i].W1, moe.experts[i].W2, moe.experts[i].W3,
            moe.experts[j].W1, moe.experts[j].W2, moe.experts[j].W3,
            h_lo, h_hi,
        )
        delta[i, j] = b
        delta[j, i] = b
    return delta


def cosine_distance_swiglu(experts) -> torch.Tensor:
    """W1+W2+W3 concatenated cosine — works for SwiGLU expert layout."""
    import torch.nn.functional as F
    N = len(experts)
    vecs = torch.stack([
        torch.cat([e.W1.detach().flatten(),
                    e.W2.detach().flatten(),
                    e.W3.detach().flatten()])
        for e in experts
    ])
    sim = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
    dist = 1.0 - sim
    dist.fill_diagonal_(0.0)
    return dist


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        plan = complete_linkage_clusters(dist, epsilon=eps)
        if len(plan.clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def free_pgd(moe, merged, h_init, n_steps, step_size, lo, hi):
    h = h_init.clone().detach().requires_grad_(True)
    best = -1.0
    for _ in range(n_steps):
        y_o, _ = moe(h)
        y_m, _ = merged(h)
        loss = -(y_o - y_m).abs().max(-1).values.sum()
        loss.backward()
        with torch.no_grad():
            h += step_size * h.grad.sign()
            h.clamp_(lo, hi)
            h.grad = None
            y_o, _ = moe(h)
            y_m, _ = merged(h)
            best = max(best, (y_o - y_m).abs().max(-1).values.max().item())
        h.requires_grad_(True)
    return best


def cert_pgd(moe, merged, plan, h_init, K, n_steps, step_size, lo, hi):
    W_g = moe.router.weight.detach()
    if not check_conditions(h_init, W_g, plan, K=K,
                             delta_threshold=0.1).all_certified.all():
        return -1.0
    h = h_init.clone().detach()
    best = -1.0
    for _ in range(n_steps):
        h_g = h.clone().requires_grad_(True)
        y_o, _ = moe(h_g)
        y_m, _ = merged(h_g)
        (-(y_o - y_m).abs().max(-1).values.sum()).backward()
        gsign = h_g.grad.sign().detach()
        accepted = False
        for s in (step_size, step_size / 2, step_size / 4):
            with torch.no_grad():
                h_t = (h + s * gsign).clamp(lo, hi)
                if check_conditions(h_t, W_g, plan, K=K,
                                     delta_threshold=0.1).all_certified.all():
                    h = h_t
                    accepted = True
                    break
        if not accepted:
            break
        with torch.no_grad():
            y_o, _ = moe(h)
            y_m, _ = merged(h)
            best = max(best, (y_o - y_m).abs().max(-1).values.max().item())
    return best


def main():
    torch.manual_seed(0)
    cfg = ToySwiGLUMoEConfig(d_model=64, d_ff=128, n_experts=16, top_k=2,
                              n_clone_pairs=2, clone_noise_std=0.005)
    moe = ToySwiGLUMoE(cfg)
    H_calib = make_calibration_data(moe, n_samples=512, seed=0)
    H_test = make_calibration_data(moe, n_samples=256, seed=999)

    print("=" * 80)
    print(f"Synthetic SwiGLU MoE: N={cfg.n_experts}, K={cfg.top_k}, "
          f"d={cfg.d_model}, d_ff={cfg.d_ff}")
    print(f"Planted clone pairs: {moe.clone_pairs}")
    print("=" * 80)

    # ============ STAGE 1 ============
    print("\n--- Stage 1: Router stability ---")
    print(f"{'r':>8} | {'%top-K':>8} | {'%stable':>8} | {'frac':>6}")
    for r in (0.0, 0.01, 0.05, 0.1, 0.2):
        in_topk, stable = stable_topk_mask(
            H_calib, moe.router.weight.detach(), r=r, K=cfg.top_k,
        )
        frac = stable.float().sum().item() / max(
            in_topk.float().sum().item(), 1)
        print(f"{r:>8.3f} | {in_topk.float().mean()*100:>7.2f}% | "
              f"{stable.float().mean()*100:>7.2f}% | {frac*100:>5.1f}%")

    # ============ STAGE 2 ============
    print("\n--- Stage 2: Pair-wise SwiGLU bound vs empirical ---")
    h_lo = H_calib.min(0).values - 0.05
    h_hi = H_calib.max(0).values + 0.05
    print("  Computing bounds for 120 pairs...", flush=True)
    d_bound = swiglu_bound_matrix(moe, h_lo, h_hi)

    # Top-3 by bound, plus clone pairs
    rows = []
    for i, j in combinations(range(cfg.n_experts), 2):
        emp = empirical_swiglu_diff(moe.experts[i], moe.experts[j], H_calib)
        is_clone = (i, j) in moe.clone_pairs
        rows.append((i, j, emp, d_bound[i, j].item(), is_clone))
    rows.sort(key=lambda r: r[3])
    print(f"\n  Smallest 5 by bound:")
    print(f"  {'pair':>6} | {'empirical':>10} | {'bound':>10} | clone?")
    for i, j, emp, b, c in rows[:5]:
        tag = "  ← CLONE" if c else ""
        print(f"  ({i:2d},{j:2d}) | {emp:>10.4f} | {b:>10.2f}{tag}")
    print(f"  Clone pairs:")
    for i, j, emp, b, c in [r for r in rows if r[4]]:
        print(f"  ({i:2d},{j:2d}) | {emp:>10.4f} | {b:>10.2f}  ← CLONE")

    # ============ STAGE 3 ============
    print("\n--- Stage 3: Cosine vs cert-aware merging ---")
    d_cosine = cosine_distance_swiglu(moe.experts)

    with torch.no_grad():
        logits = H_calib @ moe.router.weight.T
    _, topk_idx = logits.topk(cfg.top_k, dim=-1)
    freq = torch.zeros(cfg.n_experts)
    for k in range(cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H_calib.shape[0]))

    for target_n in (12, 8):
        print(f"\n  16 → {target_n}:")
        # Cosine
        eps_c = find_cutoff(d_cosine, target_n)
        plan_cos = complete_linkage_clusters(d_cosine, epsilon=eps_c)
        merged_cos = apply_merge_swiglu(moe, plan_cos, freq)

        # Cert-aware
        max_b = d_cosine[d_cosine.isfinite()].max().item() * 1.001
        plan_ca = cert_aware_greedy_merge(
            moe, H_calib, d_cosine,
            target_n_clusters=target_n,
            max_bound=max_b, delta_em=0.1, bound_weight=0.0,
        )
        merged_ca = apply_merge_swiglu(moe, plan_ca, freq)

        with torch.no_grad():
            y_o, _ = moe(H_test)
            y_cos, _ = merged_cos(H_test)
            y_ca, _ = merged_ca(H_test)
        rel_cos = (y_o - y_cos).abs().mean().item() / y_o.abs().mean().item()
        rel_ca = (y_o - y_ca).abs().mean().item() / y_o.abs().mean().item()

        m_cos = check_conditions(H_test, moe.router.weight.detach(),
                                  plan_cos, K=cfg.top_k, delta_threshold=0.1)
        m_ca = check_conditions(H_test, moe.router.weight.detach(),
                                 plan_ca, K=cfg.top_k, delta_threshold=0.1)
        print(f"    cosine    : rel_div {rel_cos*100:.1f}%, "
              f"cert {m_cos.all_certified.float().mean()*100:.1f}%")
        print(f"    cert-aware: rel_div {rel_ca*100:.1f}%, "
              f"cert {m_ca.all_certified.float().mean()*100:.1f}%")
        plans = {"cosine": (plan_cos, merged_cos),
                 "cert-aware": (plan_ca, merged_ca)}

    # ============ STAGE 4 ============
    print("\n--- Stage 4: Adversarial PGD at 16→8 ---")
    lo = H_calib.min(0).values - 0.5
    hi = H_calib.max(0).values + 0.5
    step_size = (hi - lo).mean().item() * 0.02
    n_starts = 8
    n_steps = 60
    torch.manual_seed(0)
    starts = (lo + (hi - lo)
              * torch.rand(n_starts, cfg.d_model)).unsqueeze(1)

    for name, (plan, merged) in plans.items():
        worst = max(free_pgd(moe, merged, starts[s],
                              n_steps, step_size, lo, hi)
                    for s in range(n_starts))
        print(f"    {name:11s}: free PGD worst = {worst:.4f}")

    # Cert-PGD on cert-aware
    plan_ca, merged_ca = plans["cert-aware"]
    m = check_conditions(H_calib, moe.router.weight.detach(),
                          plan_ca, K=cfg.top_k, delta_threshold=0.1)
    cert_pool = H_calib[m.all_certified]
    print(f"\n    cert-aware cert pool: {cert_pool.shape[0]}/{H_calib.shape[0]}")
    if cert_pool.shape[0] >= 4:
        n_c = min(n_starts, cert_pool.shape[0])
        worst_cp = max(
            cert_pgd(moe, merged_ca, plan_ca, cert_pool[s:s+1],
                      cfg.top_k, n_steps, step_size, lo, hi)
            for s in range(n_c)
        )
        print(f"    cert-aware cert-PGD worst = {worst_cp:.4f}")

        worst_free_cos = max(
            free_pgd(moe, plans["cosine"][1], starts[s],
                      n_steps, step_size, lo, hi)
            for s in range(n_starts)
        )
        if worst_free_cos > 0:
            red = (worst_free_cos - worst_cp) / worst_free_cos * 100
            print(f"    reduction vs cosine free PGD: {red:.1f}%")


if __name__ == "__main__":
    main()
