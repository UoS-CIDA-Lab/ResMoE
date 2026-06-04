"""Experiment 12 — Smoke test SwiGLU bound implementation.

Build a tiny synthetic SwiGLU expert, run the bound, and compare with
empirical samples. Sanity:
  - Bound must be SOUND (always ≥ empirical max).
  - Bound should DISCRIMINATE clones from non-clones (tighter for clones).
  - Sample-based numerical check on SiLU linear bounds.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.swiglu_bounds import (
    silu, silu_linear_bounds, silu_interval,
    swiglu_diff_bound_box,
)


def test_silu_bounds_soundness():
    """For random intervals, check that linear bounds enclose SiLU."""
    print("--- Test 1: SiLU linear bounds soundness ---")
    torch.manual_seed(0)
    n = 500
    # Random intervals across a wide range
    centers = torch.randn(n) * 3.0
    widths = torch.rand(n) * 2.0 + 0.1
    u_lo = centers - widths / 2
    u_hi = centers + widths / 2

    a_L, b_L, a_U, b_U = silu_linear_bounds(u_lo, u_hi)

    # Sample points in each interval
    n_samples = 30
    alphas = torch.linspace(0, 1, n_samples)
    samples = u_lo.unsqueeze(0) + alphas.unsqueeze(1) * (u_hi - u_lo).unsqueeze(0)
    phi = silu(samples)

    lower = a_L.unsqueeze(0) * samples + b_L.unsqueeze(0)
    upper = a_U.unsqueeze(0) * samples + b_U.unsqueeze(0)

    violations_lower = (phi < lower - 1e-4).sum().item()
    violations_upper = (phi > upper + 1e-4).sum().item()

    print(f"  {n} random intervals, {n_samples} samples each "
          f"({n*n_samples} total checks)")
    print(f"  Lower-bound violations: {violations_lower}")
    print(f"  Upper-bound violations: {violations_upper}")
    if violations_lower == 0 and violations_upper == 0:
        print("  PASS — bounds are sound.")
    else:
        print(f"  FAIL — bounds have {violations_lower+violations_upper} "
              f"violations.")
        # Inspect worst violation
        worst_l = (lower - phi).max().item()
        worst_u = (phi - upper).max().item()
        print(f"  Worst lower excess: {worst_l:.4f}")
        print(f"  Worst upper excess: {worst_u:.4f}")


def test_silu_interval():
    """Verify silu_interval matches numerical truth on a wide range."""
    print("\n--- Test 2: SiLU interval bound correctness ---")
    torch.manual_seed(1)
    n = 200
    centers = torch.randn(n) * 3.0
    widths = torch.rand(n) * 2.0 + 0.1
    u_lo = centers - widths / 2
    u_hi = centers + widths / 2

    phi_lo, phi_hi = silu_interval(u_lo, u_hi)

    # Sample densely
    n_samples = 100
    alphas = torch.linspace(0, 1, n_samples)
    samples = u_lo.unsqueeze(0) + alphas.unsqueeze(1) * (u_hi - u_lo).unsqueeze(0)
    phi = silu(samples)
    true_lo = phi.min(0).values
    true_hi = phi.max(0).values

    lo_err = (true_lo - phi_lo).clamp(min=0).max().item()
    hi_err = (phi_hi - true_hi).clamp(min=0).max().item()
    print(f"  Max lower-interval excess (should be small): {lo_err:.4f}")
    print(f"  Max upper-interval excess (should be small): {hi_err:.4f}")
    if lo_err > 1e-3 or hi_err > 1e-3:
        print("  WARN — interval not tight; check silu_interval logic.")
    else:
        print("  PASS — interval bounds are tight.")


def make_swiglu_expert(d_model: int, d_ff: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    s = (d_model ** -0.5)
    return (
        torch.randn(d_ff, d_model, generator=g) * s,  # W1
        torch.randn(d_ff, d_model, generator=g) * s,  # W2
        torch.randn(d_model, d_ff, generator=g) * s,  # W3
    )


def swiglu_forward(W1, W2, W3, h):
    return (silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def test_swiglu_diff_bound():
    """Build a small SwiGLU MoE, compare bound to empirical for clone vs not."""
    print("\n--- Test 3: SwiGLU difference bound on clone vs non-clone ---")
    torch.manual_seed(2)
    d_model, d_ff = 32, 64
    # Three experts: 0 and 1 are clones (small noise), 2 is random
    W1_0, W2_0, W3_0 = make_swiglu_expert(d_model, d_ff, seed=0)
    noise = 0.01
    W1_1 = W1_0 + noise * torch.randn_like(W1_0)
    W2_1 = W2_0 + noise * torch.randn_like(W2_0)
    W3_1 = W3_0 + noise * torch.randn_like(W3_0)
    W1_2, W2_2, W3_2 = make_swiglu_expert(d_model, d_ff, seed=99)

    # Calibration data
    H = torch.randn(256, d_model) * 0.5
    h_lo = H.min(0).values - 0.1
    h_hi = H.max(0).values + 0.1

    # Empirical diffs
    def emp_diff(Wa, Wb):
        with torch.no_grad():
            y_a = swiglu_forward(*Wa, H)
            y_b = swiglu_forward(*Wb, H)
        return (y_a - y_b).abs().max().item()

    e_clone = emp_diff((W1_0, W2_0, W3_0), (W1_1, W2_1, W3_1))
    e_other = emp_diff((W1_0, W2_0, W3_0), (W1_2, W2_2, W3_2))

    b_clone = swiglu_diff_bound_box(
        W1_0, W2_0, W3_0, W1_1, W2_1, W3_1, h_lo, h_hi,
    )
    b_other = swiglu_diff_bound_box(
        W1_0, W2_0, W3_0, W1_2, W2_2, W3_2, h_lo, h_hi,
    )

    print(f"  Clone pair (0,1):     empirical={e_clone:.4f}, "
          f"bound={b_clone:.4f}, ratio={b_clone/max(e_clone,1e-8):.1f}x")
    print(f"  Non-clone pair (0,2): empirical={e_other:.4f}, "
          f"bound={b_other:.4f}, ratio={b_other/max(e_other,1e-8):.1f}x")

    if b_clone < b_other:
        print("  PASS — clone bound is smaller than non-clone bound.")
    else:
        print("  FAIL — bound does not discriminate clones.")
    if b_clone >= e_clone - 1e-3 and b_other >= e_other - 1e-3:
        print("  PASS — bounds are sound (≥ empirical).")
    else:
        print("  FAIL — soundness violated.")


def main():
    test_silu_bounds_soundness()
    test_silu_interval()
    test_swiglu_diff_bound()


if __name__ == "__main__":
    main()
