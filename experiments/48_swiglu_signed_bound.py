"""Experiment 48 — Sign-preserving SwiGLU difference bound (capture W3-sum
cancellation). Tightens exp 47.

exp 47's residual slack was the final L1 step |W3| @ g_diff, which discards sign
cancellation across the 1024 hidden units in the W3 sum. Fix: split each
dominant LINEAR-in-x term into a CENTER part (exact, sign-preserving -> the
W3-sum cancellation is kept) plus a RADIUS part (small L1 remainder). The full
interval coupling between sigma(u'_k) and the (out,in) indices is O(d_out*d_ff*d)
= infeasible, so we use sigma(u'_k) = s_c +- s_r (center/radius over the box):

  D_o = sum_k W3[o,k](g_k - g'_k) + sum_k d3[o,k] g'_k
  g_k - g'_k = [sigma(u_k)-sigma(u'_k)] v_k        (A: bilinear, keep L1, ~2nd order)
             + sigma(u'_k) (d2[k].x)               (B: linear in x)
  B center  : sum_k W3[o,k] s_c_k (d2[k].x) = ((W3*s_c) @ d2)[o] . x   (EXACT, signed)
  B radius  : sum_k |W3[o,k]| s_r_k max|d2[k].x|                       (L1, small s_r)
  d3 center : sum_k d3[o,k] gc_k                                       (signed sum)
  d3 radius : sum_k |d3[o,k]| gr_k
The signed center terms aggregate W3/d2/d3 with signs -> cross-unit cancellation.

Compares McCormick vs exp47-pert vs THIS signed bound vs PGD on OLMoE quant.
Run: python3 experiments/48_swiglu_signed_bound.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import swiglu_diff_bound_box, silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
LSIG = 1.1
SILU_MIN_Z = -1.2784
SILU_MIN_V = -0.2785


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def affine_box(W, lo, hi):
    c = (lo + hi) / 2
    r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_interval(a, b):
    """[min,max] of SiLU over [a,b], elementwise (sound)."""
    sa, sb = silu(a), silu(b)
    smax = torch.maximum(sa, sb)
    smin = torch.minimum(sa, sb)
    straddle = (a <= SILU_MIN_Z) & (SILU_MIN_Z <= b)
    smin = torch.where(straddle, torch.minimum(smin, torch.full_like(smin, SILU_MIN_V)), smin)
    return smin, smax


def interval_mul(alo, ahi, blo, bhi):
    p = torch.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi], 0)
    return p.amin(0), p.amax(0)


def swiglu_signed_bound(W1, W2, W3, W1q, W2q, W3q, h_lo, h_hi):
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    c = (h_lo + h_hi) / 2
    r = (h_hi - h_lo) / 2
    # pre-activation intervals
    uq_lo, uq_hi = affine_box(W1q, h_lo, h_hi)
    v_lo, v_hi = affine_box(W2, h_lo, h_hi)
    vq_lo, vq_hi = affine_box(W2q, h_lo, h_hi)
    s_lo, s_hi = silu_interval(uq_lo, uq_hi)          # sigma(u')
    s_c, s_r = (s_lo + s_hi) / 2, (s_hi - s_lo) / 2
    # magnitudes (linear-in-x max abs over box)
    amax = lambda lo, hi: torch.maximum(lo.abs(), hi.abs())
    Bv = amax(v_lo, v_hi)
    Bd1 = (d1 @ c).abs() + d1.abs() @ r
    Bd2 = (d2 @ c).abs() + d2.abs() @ r
    # g' interval
    glo, ghi = interval_mul(s_lo, s_hi, vq_lo, vq_hi)
    gc, gr = (glo + ghi) / 2, (ghi - glo) / 2

    # B center (signed, exact linear in x): a[o,:] . x, a = (W3*s_c) @ d2
    a = (W3 * s_c.unsqueeze(0)) @ d2                  # [d_out, d]
    lin_c = a @ c
    lin_rad = a.abs() @ r
    # d3 center (signed constant)
    d3c = d3 @ gc
    # remainders (L1)
    A_term = W3.abs() @ (LSIG * Bd1 * Bv)             # A: bilinear, magnitude
    B_rem = W3.abs() @ (s_r * Bd2)
    d3_rem = d3.abs() @ gr
    R = A_term + B_rem + d3_rem
    lo = lin_c - lin_rad + d3c - R
    hi = lin_c + lin_rad + d3c + R
    return torch.maximum(lo.abs(), hi.abs()).max().item()


def swiglu_pert_bound(W1, W2, W3, W1q, W2q, W3q, h_lo, h_hi):
    """exp 47 (L1) bound, for comparison."""
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    uq_lo, uq_hi = affine_box(W1q, h_lo, h_hi)
    v_lo, v_hi = affine_box(W2, h_lo, h_hi)
    vq_lo, vq_hi = affine_box(W2q, h_lo, h_hi)
    dd1_lo, dd1_hi = affine_box(d1, h_lo, h_hi)
    dd2_lo, dd2_hi = affine_box(d2, h_lo, h_hi)
    amax = lambda lo, hi: torch.maximum(lo.abs(), hi.abs())
    Bv, Bvq = amax(v_lo, v_hi), amax(vq_lo, vq_hi)
    Bd1, Bd2 = amax(dd1_lo, dd1_hi), amax(dd2_lo, dd2_hi)
    s_lo, s_hi = silu_interval(uq_lo, uq_hi)
    Suq = torch.maximum(s_lo.abs(), s_hi.abs())
    g_diff = LSIG * Bd1 * Bv + Suq * Bd2
    gq = Suq * Bvq
    return (W3.abs() @ g_diff + d3.abs() @ gq).max().item()


def pgd_pair(Wi, Wj, h0, eps, n_steps=120):
    (W1i, W2i, W3i), (W1j, W2j, W3j) = Wi, Wj
    step = eps / 12 if eps > 0 else 0.0
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps if eps > 0 else 1):
        hg = h.clone().requires_grad_(True)
        d = (swiglu(hg, W1i, W2i, W3i) - swiglu(hg, W1j, W2j, W3j)
             ).abs().max(-1).values.sum()
        d.backward()
        with torch.no_grad():
            if eps > 0:
                h = torch.max(torch.min(h + step * hg.grad.sign(), h0 + eps), h0 - eps)
            cur = (swiglu(h, W1i, W2i, W3i) - swiglu(h, W1j, W2j, W3j)
                   ).abs().max(-1).values.max().item()
            best = max(best, cur)
    return best


def quantize_per_channel(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / scale).clamp(-qmax - 1, qmax) * scale


def main():
    print("Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float()
    H = c["H"].float()
    N = c["n_experts"]
    logits = H @ Wg.T
    torch.manual_seed(0)
    pool = torch.randperm(H.shape[0])[:20]

    for bits in (8, 4):
        W1q = torch.stack([quantize_per_channel(W1[i], bits) for i in range(N)])
        W2q = torch.stack([quantize_per_channel(W2[i], bits) for i in range(N)])
        W3q = torch.stack([quantize_per_channel(W3[i], bits) for i in range(N)])
        print(f"\n{'='*78}\n  {bits}-bit quant: PGD vs McCormick vs pert(exp47) vs SIGNED(exp48)"
              f"\n{'='*78}")
        print(f"  {'eps':>6s} {'PGD':>9s} {'McCormick':>10s} {'pert':>10s} "
              f"{'signed':>10s} {'McC×':>7s} {'pert×':>7s} {'signed×':>8s}")
        print("  " + "-" * 72)
        for eps in (0.0, 0.01, 0.05, 0.1):
            E, M, P, S = [], [], [], []
            for b in pool.tolist():
                h = H[b:b + 1]
                i = int(logits[b].argmax())
                lo, hi = (h - eps).squeeze(0), (h + eps).squeeze(0)
                Wi = (W1[i], W2[i], W3[i]); Wj = (W1q[i], W2q[i], W3q[i])
                E.append(pgd_pair(Wi, Wj, h, eps))
                M.append(swiglu_diff_bound_box(*Wi, *Wj, lo, hi))
                P.append(swiglu_pert_bound(*Wi, *Wj, lo, hi))
                S.append(swiglu_signed_bound(*Wi, *Wj, lo, hi))
            me = sorted(E)[len(E) // 2]; mm = sorted(M)[len(M) // 2]
            mp = sorted(P)[len(P) // 2]; ms = sorted(S)[len(S) // 2]
            print(f"  {eps:>6.3f} {me:>9.4f} {mm:>10.3f} {mp:>10.3f} {ms:>10.4f} "
                  f"{mm/max(me,1e-9):>6.0f}× {mp/max(me,1e-9):>6.0f}× "
                  f"{ms/max(me,1e-9):>7.1f}×")

    print(f"\n{'='*78}")
    print("  signed× should be << pert× (exp47): keeping the W3-sum sign structure")
    print("  captures cross-unit cancellation. signed must stay >= PGD (sound).")


if __name__ == "__main__":
    main()
