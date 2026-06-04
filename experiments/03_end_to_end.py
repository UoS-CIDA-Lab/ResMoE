"""Experiment 03 — End-to-end CertMerge pipeline.

Pipeline:
  Stage 1: For each candidate pair (i, j), find stable calibration
           subset (router-stable for either i or j) via closed-form check.
  Stage 2: For each pair with sufficient stable support, construct
           input region (box over stable subset) and verify pair
           equivalence with CROWN.
  Stage 3: Build merge graph from verified pairs, cluster with
           complete-linkage, merge experts + router.

Outcomes measured:
  - Number of verified pairs at various ε
  - Whether planted clones (12,13) and (14,15) are among them
  - Empirical output divergence of merged MoE vs original on test data
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
from itertools import combinations

import torch

from cert_moe.toy_moe import ToyMoE, ToyMoEConfig, make_calibration_data
from cert_moe.router_stability import stable_topk_mask
from cert_moe.expert_bounds import pair_diff_bound, empirical_pair_diff
from cert_moe.merge import complete_linkage_clusters, apply_merge
from cert_moe.theorem1_conditions import check_conditions


def build_input_region(
    H_stable: torch.Tensor,
    margin_frac: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned bounding box over stable samples, expanded by margin."""
    if H_stable.shape[0] == 0:
        # Empty: return a degenerate (vacuous) region
        d = H_stable.shape[1] if H_stable.ndim > 1 else 64
        return torch.zeros(d), torch.zeros(d)
    h_lo = H_stable.min(0).values
    h_hi = H_stable.max(0).values
    span = (h_hi - h_lo).clamp(min=1e-6)
    h_lo = h_lo - margin_frac * span
    h_hi = h_hi + margin_frac * span
    return h_lo, h_hi


def cert_merge_pipeline(
    moe: ToyMoE,
    H: torch.Tensor,
    r: float,
    epsilon: float,
    min_support: int = 5,
    verbose: bool = True,
    use_global_region: bool = True,
):
    """Run the full CertMerge pipeline. Returns merged MoE + diagnostics.

    use_global_region: if True (recommended), use the bounding box of ALL
        stable samples as the common input region for all pair bounds.
        This keeps bounds comparable across pairs. If False, per-pair
        regions are used (often introduces spurious correlation between
        bound and expert usage frequency).
    """
    N = moe.cfg.n_experts
    K = moe.cfg.top_k
    W_g = moe.router.weight.detach()

    # ============ Stage 1: router stability ============
    _, stable_mask = stable_topk_mask(H, W_g, r=r, K=K)  # [B, N]

    # Common input region: samples where ANY expert is stably in TopK
    any_stable = stable_mask.any(-1)
    H_global_stable = H[any_stable]
    if H_global_stable.shape[0] == 0:
        H_global_stable = H  # fall back to full calibration
    h_lo_global, h_hi_global = build_input_region(H_global_stable)

    # ============ Stage 2: per-pair verification ============
    delta_hat = torch.full((N, N), math.inf)
    delta_hat.fill_diagonal_(0.0)
    pair_diag = []  # for logging

    for i, j in combinations(range(N), 2):
        # Stable calibration set for pair (i, j) — counts toward support only
        relevant = (stable_mask[:, i] | stable_mask[:, j])
        n_support = relevant.sum().item()

        if n_support < min_support:
            pair_diag.append((i, j, n_support, math.inf, "low support"))
            continue

        if use_global_region:
            h_lo, h_hi = h_lo_global, h_hi_global
        else:
            H_stable_pair = H[relevant]
            h_lo, h_hi = build_input_region(H_stable_pair)

        bound = pair_diff_bound(moe.experts[i], moe.experts[j],
                                 h_lo, h_hi, method="crown")
        delta_hat[i, j] = bound
        delta_hat[j, i] = bound
        pair_diag.append((i, j, n_support, bound, "verified"))

    # ============ Stage 3: cluster + merge ============
    plan = complete_linkage_clusters(delta_hat, epsilon=epsilon)

    if verbose:
        verified_pairs = [(i, j, b) for i, j, _, b, st in pair_diag
                           if st == "verified" and b <= epsilon]
        print(f"\nStage 2: verified pairs at ε={epsilon}: {len(verified_pairs)}")
        for i, j, b in sorted(verified_pairs, key=lambda x: x[2])[:8]:
            is_clone = (i, j) in moe.clone_pairs or (j, i) in moe.clone_pairs
            tag = "  ← CLONE" if is_clone else ""
            print(f"  ({i:2d},{j:2d}) δ̂ = {b:.4f}{tag}")
        print(f"\nStage 3: clusters ({len(plan.clusters)} clusters from {N} experts):")
        for k, members in enumerate(plan.clusters):
            if len(members) > 1:
                print(f"  cluster {k}: {members}")

    # ============ Build merged MoE ============
    # Routing frequency from calibration
    with torch.no_grad():
        logits = H @ W_g.T
        _, topk_idx = logits.topk(K, dim=-1)
    freq = torch.zeros(N)
    for k in range(K):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))

    merged_moe = apply_merge(moe, plan, freq)

    return merged_moe, plan, delta_hat


def measure_output_divergence(
    moe: ToyMoE, merged_moe: ToyMoE, H: torch.Tensor,
    plan=None,
    delta_em: float = 0.1,
) -> dict:
    """Output divergence with per-condition bucketing (Theorem 1).

    Buckets tokens by which subset of {MC, TS, EM} they satisfy.
    Certified = all three hold → Theorem 1 bound applies.
    """
    with torch.no_grad():
        y_orig, _ = moe(H)
        y_merged, _ = merged_moe(H)
    diff = (y_orig - y_merged).abs()
    diff_per_token = diff.max(-1).values  # [B]

    out = {
        "max_per_token": diff.max(-1).values.mean().item(),
        "mean": diff.mean().item(),
        "max_overall": diff.max().item(),
        "relative_to_output_norm": (
            diff.mean().item() / y_orig.abs().mean().item()
        ),
    }
    if plan is not None:
        K = moe.cfg.top_k
        masks = check_conditions(
            H, moe.router.weight.detach(), plan, K=K,
            delta_threshold=delta_em,
        )
        out["mc_frac"] = masks.mc.float().mean().item()
        out["ts_frac"] = masks.ts.float().mean().item()
        out["em_frac"] = masks.em.float().mean().item()
        out["certified_frac"] = masks.all_certified.float().mean().item()
        out["xi_mean"] = masks.xi.mean().item()
        out["xi_max"] = masks.xi.max().item()

        cert = masks.all_certified
        if cert.any():
            out["certified_L_inf_mean"] = diff_per_token[cert].mean().item()
            out["certified_L_inf_max"]  = diff_per_token[cert].max().item()
        if (~cert).any():
            out["uncertified_L_inf_mean"] = diff_per_token[~cert].mean().item()
            out["uncertified_L_inf_max"]  = diff_per_token[~cert].max().item()
    return out


def main():
    torch.manual_seed(0)
    cfg = ToyMoEConfig(d_model=64, d_ff=128, n_experts=16, top_k=2,
                       n_clone_pairs=2, clone_noise_std=0.005)
    moe = ToyMoE(cfg)
    H_calib = make_calibration_data(moe, n_samples=512, seed=0)
    H_test = make_calibration_data(moe, n_samples=256, seed=999)

    print("=" * 70)
    print("CertMerge pipeline on toy MoE")
    print(f"  N={cfg.n_experts}, top-K={cfg.top_k}")
    print(f"  Planted clone pairs: {moe.clone_pairs}")
    print("=" * 70)

    # Try a few (r, ε) configurations
    # ε=72.0 is the "discriminating" range — between clone bound (~70)
    # and lowest non-clone (~74) — should isolate clones cleanly.
    configs = [
        (0.01, 72.0),   # discriminating — should catch only clones
        (0.01, 78.0),   # loose — catches some non-clones
        (0.01, 90.0),   # very loose — merges everything
    ]

    for r, eps in configs:
        print(f"\n{'='*70}")
        print(f"Config: r={r}, ε={eps}")
        print("=" * 70)
        merged, plan, _ = cert_merge_pipeline(
            moe, H_calib, r=r, epsilon=eps, verbose=True
        )

        # Measure divergence on TEST data (not used for verification)
        div = measure_output_divergence(moe, merged, H_test, plan=plan)
        print(f"\nOutput divergence on test data:")
        print(f"  overall mean |y - ŷ|          : {div['mean']:.4f}")
        print(f"  overall max L∞ per token mean : {div['max_per_token']:.4f}")
        print(f"  relative to ||y_orig|| mean   : "
              f"{div['relative_to_output_norm']*100:.2f}%")
        if "certified_frac" in div:
            print(f"\n  Theorem 1 condition pass rates:")
            print(f"    MC          : {div['mc_frac']*100:5.1f}%")
            print(f"    TS          : {div['ts_frac']*100:5.1f}%")
            print(f"    EM (δ=0.1)  : {div['em_frac']*100:5.1f}%")
            print(f"    ALL THREE   : {div['certified_frac']*100:5.1f}%  ← Theorem 1 applies here")
            print(f"    Ξ(h)        : mean={div['xi_mean']:.4f}, "
                  f"max={div['xi_max']:.4f}")
            if "certified_L_inf_mean" in div:
                print(f"\n  L∞ divergence by bucket:")
                print(f"    certified   : "
                      f"mean={div['certified_L_inf_mean']:.4f}, "
                      f"max={div['certified_L_inf_max']:.4f}")
            if "uncertified_L_inf_mean" in div:
                print(f"    uncertified : "
                      f"mean={div['uncertified_L_inf_mean']:.4f}, "
                      f"max={div['uncertified_L_inf_max']:.4f}")
                if "certified_L_inf_mean" in div:
                    ratio = (div['uncertified_L_inf_mean']
                             / max(div['certified_L_inf_mean'], 1e-8))
                    print(f"    ratio uncert/cert : {ratio:.2f}x")
        print(f"\n  experts: {cfg.n_experts} → {len(plan.clusters)} "
              f"(compression ratio {cfg.n_experts/len(plan.clusters):.2f}x)")

    # Baseline: random merging at same compression ratio
    print(f"\n{'='*70}")
    print("Baseline: RANDOM merging at compression 16→14 (matches tightest ε)")
    print("=" * 70)
    torch.manual_seed(42)
    # Pick 2 random pairs to merge
    all_pairs = list(combinations(range(16), 2))
    perm = torch.randperm(len(all_pairs))[:2].tolist()
    rand_pairs = [all_pairs[p] for p in perm]
    # Build delta_hat matrix that "verifies" these random pairs
    rand_delta = torch.full((16, 16), math.inf)
    rand_delta.fill_diagonal_(0.0)
    for i, j in rand_pairs:
        rand_delta[i, j] = 0.0
        rand_delta[j, i] = 0.0
    rand_plan = complete_linkage_clusters(rand_delta, epsilon=1.0)
    print(f"Random merge pairs: {rand_pairs}")
    with torch.no_grad():
        logits = H_calib @ moe.router.weight.T
        _, topk_idx = logits.topk(2, dim=-1)
    freq = torch.zeros(16)
    for k in range(2):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H_calib.shape[0]))
    rand_merged = apply_merge(moe, rand_plan, freq)
    div = measure_output_divergence(moe, rand_merged, H_test, plan=rand_plan)
    print(f"\nOutput divergence (random merge):")
    print(f"  mean |y_orig - y_merged|       : {div['mean']:.4f}")
    print(f"  max per-token L∞              : {div['max_per_token']:.4f}")
    print(f"  relative to ||y_orig|| mean   : "
          f"{div['relative_to_output_norm']*100:.2f}%")
    if "certified_frac" in div:
        print(f"\n  Theorem 1 pass rates:")
        print(f"    MC          : {div['mc_frac']*100:5.1f}%")
        print(f"    TS          : {div['ts_frac']*100:5.1f}%")
        print(f"    EM          : {div['em_frac']*100:5.1f}%")
        print(f"    ALL THREE   : {div['certified_frac']*100:5.1f}%")
        if "certified_L_inf_mean" in div:
            print(f"    certified L∞ mean : "
                  f"{div['certified_L_inf_mean']:.4f}")
        if "uncertified_L_inf_mean" in div:
            print(f"    uncertified L∞ mean : "
                  f"{div['uncertified_L_inf_mean']:.4f}")


if __name__ == "__main__":
    main()
