"""Experiment 05 — Adversarial worst-case divergence search.

Question: for each merged MoE (ours + each baseline), find inputs h that
maximize ||M(h) - M̂(h)||_∞ via PGD-style search. Compare:

  - heuristic-merged MoE: worst-case can be very large (no guarantee)
  - certified-merged MoE: worst-case bounded on certified region;
    UN-certified region may also be large, but we can REJECT those
    inputs at deployment time.

Key paper claim: heuristic methods provide no defense against worst-
case inputs; our framework lets you detect and reject them.

We run PGD in two modes:
  (a) FREE search: start from random h, no constraint on certifiability
  (b) CERTIFIED search: project onto certified region (for our method
      only — heuristics have no such region)

For (a), all methods are evaluated; for (b), only ours.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from itertools import combinations
import math

import torch

from cert_moe.toy_moe import ToyMoE, ToyMoEConfig, make_calibration_data
from cert_moe.router_stability import stable_topk_mask
from cert_moe.expert_bounds import (
    pair_diff_bound, zonotope_from_data, alpha_crown_zonotope_bound,
)
from cert_moe.merge import (
    complete_linkage_clusters, apply_merge, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions
from cert_moe.baselines import (
    cosine_distance_matrix, mc_smoe_distance_matrix,
    hc_smoe_distance_matrix,
)


def certified_distance_matrix(
    moe, H, r=0.01, K=2,
    region_type: str = "box", pca_k: int | None = None,
):
    N = moe.cfg.n_experts
    W_g = moe.router.weight.detach()
    _, stable_mask = stable_topk_mask(H, W_g, r=r, K=K)
    any_stable = stable_mask.any(-1)
    H_stable = H[any_stable] if any_stable.any() else H
    delta = torch.full((N, N), math.inf)
    delta.fill_diagonal_(0.0)
    if region_type == "box":
        h_lo = H_stable.min(0).values - 0.05
        h_hi = H_stable.max(0).values + 0.05
        for i, j in combinations(range(N), 2):
            b = pair_diff_bound(moe.experts[i], moe.experts[j],
                                h_lo, h_hi, method="alpha_crown",
                                alpha_iters=20)
            delta[i, j] = b
            delta[j, i] = b
    elif region_type == "zonotope":
        center, V, alpha = zonotope_from_data(
            H_stable, n_components=pca_k, margin=0.05
        )
        for i, j in combinations(range(N), 2):
            b = alpha_crown_zonotope_bound(
                moe.experts[i], moe.experts[j],
                center, V, alpha, n_iters=20,
            )
            delta[i, j] = b
            delta[j, i] = b
    return delta


def find_cutoff_for_compression(dist, target_n_clusters):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        plan = complete_linkage_clusters(dist, epsilon=eps)
        if len(plan.clusters) <= target_n_clusters:
            return eps
    return flat[-1] if flat else 1.0


def pgd_divergence_attack(
    moe, merged_moe, h_init: torch.Tensor,
    n_steps: int = 200, step_size: float = 0.05,
    bound_box: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, float]:
    """Project Gradient Descent to maximize ||M(h) - M̂(h)||_∞.

    Returns (worst_h, worst_divergence).
    """
    h = h_init.clone().detach().requires_grad_(True)
    best_div = -1.0
    best_h = h_init.clone()

    for _ in range(n_steps):
        y_orig, _ = moe(h)
        y_merged, _ = merged_moe(h)
        diff = (y_orig - y_merged).abs()
        loss = -diff.max(-1).values.sum()   # negative because we maximize
        loss.backward()
        with torch.no_grad():
            h += step_size * h.grad.sign()
            if bound_box is not None:
                lo, hi = bound_box
                h.clamp_(lo, hi)
            h.grad = None

            # Track best
            y_orig, _ = moe(h)
            y_merged, _ = merged_moe(h)
            cur = (y_orig - y_merged).abs().max(-1).values.max().item()
            if cur > best_div:
                best_div = cur
                best_h = h.clone()

        h.requires_grad_(True)

    return best_h, best_div


def run_cert_pgd(moe, merged_ours, plan_ours, top_k,
                 h_lo_box, h_hi_box, n_attack_starts, n_attack_steps):
    """Run certified-region projected PGD and print a one-block summary."""
    # Find seed certified inputs
    H_pool = make_calibration_data(moe, n_samples=2048, seed=2024)
    masks = check_conditions(H_pool, moe.router.weight.detach(),
                              plan_ours, K=top_k,
                              delta_threshold=0.1)
    cert_pool = H_pool[masks.all_certified]
    n_cert_avail = cert_pool.shape[0]
    if n_cert_avail < 5:
        print(f"      Only {n_cert_avail} certified seeds — skipping.")
        return
    n_seeds = min(n_attack_starts, n_cert_avail)
    worst = 0.0
    base_max = 0.0
    total_cert_steps = 0
    for s in range(n_seeds):
        h0 = cert_pool[s:s + 1]
        with torch.no_grad():
            y_o, _ = moe(h0)
            y_m, _ = merged_ours(h0)
            base = (y_o - y_m).abs().max(-1).values.max().item()
        base_max = max(base_max, base)
        _, div, cert_steps = pgd_certified_region_attack(
            moe, merged_ours, plan_ours, h0,
            bound_box=(h_lo_box, h_hi_box),
            K=top_k,
            n_steps=n_attack_steps, step_size=0.05,
        )
        worst = max(worst, div)
        total_cert_steps += cert_steps
    avg_cert_steps = total_cert_steps / n_seeds
    print(f"      Seeds          : {n_seeds} certified starts "
          f"(pool size {n_cert_avail})")
    print(f"      Baseline div   : {base_max:.4f}")
    print(f"      Cert-PGD worst : {worst:.4f}")
    print(f"      Avg cert steps : {avg_cert_steps:.1f}/{n_attack_steps}")


def pgd_certified_region_attack(
    moe, merged_moe, plan,
    h_init: torch.Tensor,
    bound_box: tuple[torch.Tensor, torch.Tensor],
    K: int,
    n_steps: int = 100, step_size: float = 0.05,
    delta_em: float = 0.1,
) -> tuple[torch.Tensor, float, int]:
    """PGD that stays within Theorem 1's certified region.

    At each step, after the gradient step, check (MC ∧ TS ∧ EM). If the
    new point falls outside the certified region, back off to a smaller
    step. If no certified step is possible, terminate early.

    Returns (worst_h, worst_div, n_certified_steps).
    """
    W_g = moe.router.weight.detach()

    # Verify initial point is certified
    masks0 = check_conditions(h_init, W_g, plan, K=K, delta_threshold=delta_em)
    if not masks0.all_certified.all():
        return h_init, -1.0, 0

    h = h_init.clone().detach()
    best_div = -1.0
    best_h = h.clone()
    n_cert_steps = 0

    for _ in range(n_steps):
        h_g = h.clone().requires_grad_(True)
        y_orig, _ = moe(h_g)
        y_merged, _ = merged_moe(h_g)
        diff = (y_orig - y_merged).abs()
        loss = -diff.max(-1).values.sum()
        loss.backward()
        grad_sign = h_g.grad.sign().detach()

        # Try step with backoff if cert violated
        accepted = False
        for s in (step_size, step_size / 2, step_size / 4, step_size / 8):
            with torch.no_grad():
                h_trial = (h + s * grad_sign).clamp(bound_box[0], bound_box[1])
                masks = check_conditions(h_trial, W_g, plan, K=K,
                                          delta_threshold=delta_em)
                if masks.all_certified.all():
                    h = h_trial
                    n_cert_steps += 1
                    accepted = True
                    break

        if not accepted:
            break  # cannot proceed without violating cert

        # Track best on h
        with torch.no_grad():
            y_orig, _ = moe(h)
            y_merged, _ = merged_moe(h)
            cur = (y_orig - y_merged).abs().max(-1).values.max().item()
            if cur > best_div:
                best_div = cur
                best_h = h.clone()

    return best_h, best_div, n_cert_steps


def main():
    torch.manual_seed(0)
    cfg = ToyMoEConfig(d_model=64, d_ff=128, n_experts=16, top_k=2,
                       n_clone_pairs=2, clone_noise_std=0.005)
    moe = ToyMoE(cfg)
    H_calib = make_calibration_data(moe, n_samples=512, seed=0)

    # Bounding box from calibration — same domain for fair attack
    h_lo_box = H_calib.min(0).values - 0.5
    h_hi_box = H_calib.max(0).values + 0.5
    box_diag = (h_hi_box - h_lo_box).norm().item()
    print(f"Attack box: per-dim span mean = "
          f"{(h_hi_box - h_lo_box).mean().item():.3f}, "
          f"diagonal {box_diag:.3f}")

    # Frequency for merge weights
    logits = H_calib @ moe.router.weight.T
    _, topk_idx = logits.topk(cfg.top_k, dim=-1)
    freq = torch.zeros(cfg.n_experts)
    for k in range(cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H_calib.shape[0]))

    print("\nBuilding distance matrices...")
    d_cert_box = certified_distance_matrix(moe, H_calib, region_type="box")
    d_cert_zono = certified_distance_matrix(
        moe, H_calib, region_type="zonotope", pca_k=32,
    )
    d_cos = cosine_distance_matrix(moe.experts)
    d_mc = mc_smoe_distance_matrix(moe.experts, H_calib,
                                    moe.router.weight.detach(), K=2)
    d_hc = hc_smoe_distance_matrix(moe.experts, H_calib)
    methods = {
        "cert-box":  d_cert_box,
        "cert-zono": d_cert_zono,
        "cosine":    d_cos,
        "mc_smoe":   d_mc,
        "hc_smoe":   d_hc,
    }

    # Compression sweep
    target_ns = [14, 12, 10]
    n_attack_starts = 20    # PGD starts per method
    n_attack_steps  = 150

    for target_n in target_ns:
        print(f"\n{'='*88}")
        print(f"Compression 16 → {target_n}")
        print(f"{'='*88}")

        merged_models = {}
        plans = {}
        for name, dist in methods.items():
            eps = find_cutoff_for_compression(dist, target_n)
            plans[name] = complete_linkage_clusters(dist, epsilon=eps)
            merged_models[name] = apply_merge(moe, plans[name], freq)

        # Add cert-aware variants using certified distances
        for src_name in ("cert-box", "cert-zono"):
            label = f"cert-aware({src_name})"
            dist = methods[src_name]
            max_b = dist[dist.isfinite()].max().item() * 1.001
            plans[label] = cert_aware_greedy_merge(
                moe, H_calib, dist,
                target_n_clusters=target_n,
                max_bound=max_b,
                delta_em=0.1,
                bound_weight=0.0,
            )
            if len(plans[label].clusters) > target_n:
                # could not reach target — skip
                del plans[label]
                continue
            merged_models[label] = apply_merge(moe, plans[label], freq)

        # Attack each merged model
        print(f"\n{'method':20s} | {'avg-case div':>12s} | "
              f"{'worst-case div':>14s} | {'amplification':>13s}")
        print("-" * 85)
        for name, merged in merged_models.items():
            # Average-case baseline
            with torch.no_grad():
                H_test = make_calibration_data(moe, n_samples=128, seed=999)
                y_o, _ = moe(H_test)
                y_m, _ = merged(H_test)
                avg_div = (y_o - y_m).abs().max(-1).values.mean().item()

            # Worst-case via PGD
            worst = 0.0
            for s in range(n_attack_starts):
                torch.manual_seed(1000 + s)
                h0 = (h_lo_box + (h_hi_box - h_lo_box)
                      * torch.rand(1, cfg.d_model)).requires_grad_(True)
                _, div = pgd_divergence_attack(
                    moe, merged, h0,
                    n_steps=n_attack_steps,
                    step_size=0.05,
                    bound_box=(h_lo_box, h_hi_box),
                )
                if div > worst:
                    worst = div
            print(f"  {name:20s} | {avg_div:>12.4f} | {worst:>14.4f} | "
                  f"{worst/max(avg_div, 1e-8):>12.2f}x")

        # ---------- Certified-region projected PGD attack ----------
        print(f"\n  Certified-region projected PGD:")
        for our_name in ("cert-box", "cert-zono",
                         "cert-aware(cert-box)", "cert-aware(cert-zono)"):
            if our_name not in merged_models:
                continue
            print(f"    [{our_name}]")
            run_cert_pgd(
                moe, merged_models[our_name], plans[our_name], cfg.top_k,
                h_lo_box, h_hi_box, n_attack_starts, n_attack_steps,
            )

    print(f"\n{'='*88}")
    print("Takeaway:")
    print("  worst-case (PGD) divergence shows how badly each method can be")
    print("  pushed under adversarial input search. Heuristics have no guard.")
    print("  Our framework's edge: on CERTIFIED inputs, divergence stays")
    print("  bounded by Theorem 1 even when adversarial probing tries to")
    print("  push it higher — and we can EXCLUDE non-certified inputs.")
    print(f"{'='*88}")


if __name__ == "__main__":
    main()
