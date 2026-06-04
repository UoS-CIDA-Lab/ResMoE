"""Experiment 04 — Direct comparison: certified vs heuristic baselines.

For each method, build a distance matrix, run complete-linkage with a
cutoff chosen to give the same compression ratio, then measure:
  (A) ID divergence on test data
  (B) OOD divergence (shifted-mean / shifted-variance Gaussian)
  (C) Theorem 1 condition pass rates (for all methods — applies to any
      merge plan, not just certified ones)
  (D) Whether planted clone pairs are correctly merged

Compression ratios swept: 16 → {14, 12, 10, 8} (paired with appropriate
cutoffs per method).
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


# --------------------------------------------------------------------------- #
#  OOD data generation                                                         #
# --------------------------------------------------------------------------- #

def make_ood_data(moe, n_samples=256, shift_kind="mean", seed=999):
    """Generate distribution-shifted hidden states.

    - mean: shift centers by a global offset
    - variance: scale noise up
    - rotation: random orthogonal transform of in-distribution centers
    """
    g = torch.Generator().manual_seed(seed)
    d = moe.cfg.d_model
    N = moe.cfg.n_experts
    centers = torch.randn(N, d, generator=g) * 0.5
    idx = torch.randint(0, N, (n_samples,), generator=g)

    if shift_kind == "mean":
        H = centers[idx] + 0.5 + 0.3 * torch.randn(n_samples, d, generator=g)
    elif shift_kind == "variance":
        H = centers[idx] + 1.0 * torch.randn(n_samples, d, generator=g)
    elif shift_kind == "rotation":
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        H = (centers[idx] + 0.3 * torch.randn(n_samples, d, generator=g)) @ Q
    else:
        raise ValueError(f"Unknown shift: {shift_kind}")
    return H


# --------------------------------------------------------------------------- #
#  Distance matrix builders                                                    #
# --------------------------------------------------------------------------- #

def certified_distance_matrix(
    moe, H, r=0.01, K=2, min_support=5,
    bound_method="alpha_crown",
    region_type="box",
    pca_k: int | None = None,
):
    """Our certified distance: pairwise verified bound on common stable region.

    region_type: 'box' (bounding box) or 'zonotope' (PCA-based).
    pca_k:       number of PCA components for zonotope (None = full d).
    """
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
                                h_lo, h_hi, method=bound_method,
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
    else:
        raise ValueError(f"Unknown region_type: {region_type}")
    return delta


def find_cutoff_for_compression(
    dist: torch.Tensor, target_n_clusters: int,
) -> float:
    """Binary search for the cutoff producing exactly `target_n_clusters`."""
    flat = dist[dist.isfinite()].flatten().tolist()
    flat = sorted(set(flat))
    # Try cutoffs from each candidate
    best_eps = None
    for eps in flat:
        plan = complete_linkage_clusters(dist, epsilon=eps)
        if len(plan.clusters) <= target_n_clusters:
            best_eps = eps
            break
    if best_eps is None:
        best_eps = flat[-1] if flat else 1.0
    return best_eps


# --------------------------------------------------------------------------- #
#  Evaluation                                                                 #
# --------------------------------------------------------------------------- #

def measure(moe, merged, H, plan, label):
    """Measure divergence + certified fraction on data H."""
    with torch.no_grad():
        y_orig, _ = moe(H)
        y_merged, _ = merged(H)
    diff = (y_orig - y_merged).abs()
    out = {
        "label": label,
        "div_mean": diff.mean().item(),
        "div_max": diff.max(-1).values.mean().item(),
        "div_rel": diff.mean().item() / y_orig.abs().mean().item(),
    }
    masks = check_conditions(
        H, moe.router.weight.detach(), plan, K=moe.cfg.top_k,
        delta_threshold=0.1,
    )
    out["cert_frac"] = masks.all_certified.float().mean().item()
    return out


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def main():
    torch.manual_seed(0)
    cfg = ToyMoEConfig(d_model=64, d_ff=128, n_experts=16, top_k=2,
                       n_clone_pairs=2, clone_noise_std=0.005)
    moe = ToyMoE(cfg)
    H_calib = make_calibration_data(moe, n_samples=512, seed=0)
    H_id_test = make_calibration_data(moe, n_samples=256, seed=999)
    H_ood_mean = make_ood_data(moe, n_samples=256, shift_kind="mean", seed=1)
    H_ood_var = make_ood_data(moe, n_samples=256, shift_kind="variance", seed=2)

    print("=" * 88)
    print(f"Baseline comparison: N={cfg.n_experts}, top-K={cfg.top_k}, "
          f"planted clones {moe.clone_pairs}")
    print("=" * 88)

    # Compute distance matrices once per method
    print("\nComputing distance matrices...")
    print("  certified-box (α-CROWN) ...", flush=True, end=" ")
    d_cert_box = certified_distance_matrix(moe, H_calib, region_type="box")
    print("done")
    print("  certified-zono (α-CROWN + PCA zonotope) ...", end=" ", flush=True)
    d_cert_zono = certified_distance_matrix(
        moe, H_calib, region_type="zonotope", pca_k=32,
    )
    print("done")
    print("  cosine ...", end=" ", flush=True)
    d_cos = cosine_distance_matrix(moe.experts)
    print("done")
    print("  mc_smoe ...", end=" ", flush=True)
    d_mc = mc_smoe_distance_matrix(
        moe.experts, H_calib, moe.router.weight.detach(), K=2
    )
    print("done")
    print("  hc_smoe ...", end=" ", flush=True)
    d_hc = hc_smoe_distance_matrix(moe.experts, H_calib)
    print("done")

    methods = {
        "cert-box":  d_cert_box,
        "cert-zono": d_cert_zono,
        "cosine":    d_cos,
        "mc_smoe":   d_mc,
        "hc_smoe":   d_hc,
    }

    # ---------- Clone identification: top-2 of each distance ----------
    print("\n--- Top-2 closest pairs per method (ground truth: clones) ---")
    for name, dist in methods.items():
        flat = []
        for i, j in combinations(range(cfg.n_experts), 2):
            d_val = dist[i, j].item()
            if math.isfinite(d_val):
                flat.append(((i, j), d_val))
        flat.sort(key=lambda x: x[1])
        top2 = flat[:2]
        clones_found = sum(
            1 for (i, j), _ in top2
            if (i, j) in moe.clone_pairs or (j, i) in moe.clone_pairs
        )
        print(f"  {name:12s}: top-2 = "
              + ", ".join(f"{p} (d={d:.3f})" for p, d in top2)
              + f"  [clones in top-2: {clones_found}/2]")

    # ---------- Compression sweep ----------
    target_ns = [14, 12, 10, 8]
    print(f"\n{'='*88}")
    print(f"Compression sweep: N=16 → {target_ns}")
    print(f"{'='*88}")

    # Frequency for log-sum-exp merge weights
    logits = H_calib @ moe.router.weight.T
    _, topk_idx = logits.topk(cfg.top_k, dim=-1)
    freq = torch.zeros(cfg.n_experts)
    for k in range(cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H_calib.shape[0]))

    for target_n in target_ns:
        print(f"\n  ── Compression 16 → {target_n} ─" + "─" * 60)
        header = f"{'method':16s} | {'ID div':>9s} | {'OOD-m div':>9s} | "
        header += f"{'OOD-v div':>9s} | {'cert%':>6s} | {'OOD/ID':>7s}"
        print(header)
        print("-" * 88)
        for name, dist in methods.items():
            eps = find_cutoff_for_compression(dist, target_n)
            plan = complete_linkage_clusters(dist, epsilon=eps)
            merged = apply_merge(moe, plan, freq)

            r_id = measure(moe, merged, H_id_test, plan, "ID")
            r_om = measure(moe, merged, H_ood_mean, plan, "OOD-m")
            r_ov = measure(moe, merged, H_ood_var, plan, "OOD-v")
            ood_id_ratio = r_om["div_mean"] / max(r_id["div_mean"], 1e-8)
            print(
                f"  {name:16s} | {r_id['div_mean']:>9.4f} | "
                f"{r_om['div_mean']:>9.4f} | {r_ov['div_mean']:>9.4f} | "
                f"{r_id['cert_frac']*100:>5.1f}% | {ood_id_ratio:>7.2f}x"
            )

        # Cert-aware variants (on the two certified distance matrices)
        for src_name in ("cert-box", "cert-zono"):
            dist = methods[src_name]
            # Use max finite value as bound threshold (allow all in-range)
            max_b = dist[dist.isfinite()].max().item() * 1.001
            plan = cert_aware_greedy_merge(
                moe, H_calib, dist,
                target_n_clusters=target_n,
                max_bound=max_b,
                delta_em=0.1,
                bound_weight=0.0,
            )
            if len(plan.clusters) > target_n:
                continue  # could not reach target
            merged = apply_merge(moe, plan, freq)
            r_id = measure(moe, merged, H_id_test, plan, "ID")
            r_om = measure(moe, merged, H_ood_mean, plan, "OOD-m")
            r_ov = measure(moe, merged, H_ood_var, plan, "OOD-v")
            ood_id_ratio = r_om["div_mean"] / max(r_id["div_mean"], 1e-8)
            label = f"cert-aware({src_name})"
            print(
                f"  {label:16s} | {r_id['div_mean']:>9.4f} | "
                f"{r_om['div_mean']:>9.4f} | {r_ov['div_mean']:>9.4f} | "
                f"{r_id['cert_frac']*100:>5.1f}% | {ood_id_ratio:>7.2f}x"
            )

    print("\n" + "=" * 88)
    print("Reading the table:")
    print("  ID div       : divergence on in-distribution test set (lower is better)")
    print("  OOD-m div    : divergence under MEAN-shifted distribution")
    print("  OOD-v div    : divergence under VARIANCE-shifted distribution")
    print("  cert%        : Theorem 1 certified fraction (on ID test)")
    print("  OOD/ID       : degradation ratio (lower means more robust)")
    print("=" * 88)


if __name__ == "__main__":
    main()
