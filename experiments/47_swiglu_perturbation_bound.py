"""Experiment 47 — A tight SwiGLU PERTURBATION bound that exploits E ~= E'.

exp 45 showed swiglu_diff_bound_box (McCormick) is ~1689x loose over an eps-ball
for quantization, because it relaxes each SiLU*v term over the input region and
only cancels [W3, -W3'] AFTER (the slack does not cancel). For routing-orthogonal
compression E' = E + perturbation (quantization: W' = W - dW), we want a bound
that SCALES with dW so it is tight when dW is small.

Derivation (sound). With u=W1 x, v=W2 x, g=sigma(u)*v, E=W3 g, and
W1'=W1-d1, W2'=W2-d2, W3'=W3-d3:
  E - E' = W3 (g - g') + d3 g'                                       (W3' = W3 - d3)
  g - g' = [sigma(u) - sigma(u')] * v  +  sigma(u') * (v - v')
         , |sigma(u)-sigma(u')| <= Lsig |u - u'| = Lsig |d1 x|       (MVT, Lsig=1.1)
         , v - v' = d2 x
So per coordinate, over a box [h_lo,h_hi] (interval arithmetic for each magnitude):
  |g_k - g'_k| <= Lsig * |d1_k . x|max * |v_k|max  +  |sigma(u'_k)|max * |d2_k . x|max
  |g'_k|       <= |sigma(u'_k)|max * |v'_k|max
  |E_o - E'_o| <= sum_k |W3[o,k]| |g_k-g'_k|  +  sum_k |d3[o,k]| |g'_k|
This is sound and proportional to (d1,d2,d3) -> tight for small quant steps.

Compares, for OLMoE quantization over eps-balls: McCormick bound vs this
perturbation bound vs PGD (empirical). Run: python3 experiments/47_...py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import swiglu_diff_bound_box, silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
LSIG = 1.1   # global Lipschitz bound for SiLU (sup|sigma'| ~ 1.0998)


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def affine_box(W, lo, hi):
    """Interval of W @ x over box [lo,hi]: returns (lo_out, hi_out) per row."""
    c = (lo + hi) / 2
    r = (hi - lo) / 2
    Wc = W @ c
    Wr = W.abs() @ r
    return Wc - Wr, Wc + Wr


def silu_absmax(a, b):
    """max |SiLU(z)| for z in [a,b], elementwise. SiLU has a local min at
    z*=-1.2784 (value -0.2785); elsewhere monotone-ish so endpoints dominate."""
    zmin = torch.full_like(a, -1.2784)
    cand = torch.stack([silu(a), silu(b)], dim=0).abs()
    m = cand.amax(0)
    contains = (a <= zmin) & (zmin <= b)
    m = torch.where(contains, torch.maximum(m, torch.full_like(m, 0.2785)), m)
    return m


def swiglu_pert_bound(W1, W2, W3, W1q, W2q, W3q, h_lo, h_hi):
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    # magnitudes over the box
    u_lo, u_hi = affine_box(W1, h_lo, h_hi)            # u (orig) — unused mag
    uq_lo, uq_hi = affine_box(W1q, h_lo, h_hi)         # u'
    v_lo, v_hi = affine_box(W2, h_lo, h_hi)
    vq_lo, vq_hi = affine_box(W2q, h_lo, h_hi)
    dd1_lo, dd1_hi = affine_box(d1, h_lo, h_hi)
    dd2_lo, dd2_hi = affine_box(d2, h_lo, h_hi)
    amax = lambda lo, hi: torch.maximum(lo.abs(), hi.abs())
    Bv = amax(v_lo, v_hi)
    Bvq = amax(vq_lo, vq_hi)
    Bd1 = amax(dd1_lo, dd1_hi)
    Bd2 = amax(dd2_lo, dd2_hi)
    Suq = silu_absmax(uq_lo, uq_hi)
    g_diff = LSIG * Bd1 * Bv + Suq * Bd2               # |g_k - g'_k| bound
    gq = Suq * Bvq                                     # |g'_k| bound
    out = W3.abs() @ g_diff + d3.abs() @ gq            # per output coord
    return out.max().item()


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
                h = torch.max(torch.min(h + step * hg.grad.sign(), h0 + eps),
                              h0 - eps)
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

    # sanity: perturbation bound must be SOUND (>= PGD empirical) at a point too
    torch.manual_seed(0)
    pool = torch.randperm(H.shape[0])[:20]

    for bits in (8, 4):
        W1q = torch.stack([quantize_per_channel(W1[i], bits) for i in range(N)])
        W2q = torch.stack([quantize_per_channel(W2[i], bits) for i in range(N)])
        W3q = torch.stack([quantize_per_channel(W3[i], bits) for i in range(N)])
        print(f"\n{'='*72}\n  {bits}-bit quant: McCormick vs PERTURBATION bound vs PGD"
              f"\n{'='*72}")
        print(f"  {'eps':>6s} {'PGD emp':>9s} {'McCormick':>11s} {'pert-bound':>11s} "
              f"{'McC loose':>10s} {'pert loose':>11s}")
        print("  " + "-" * 64)
        for eps in (0.0, 0.01, 0.05, 0.1):
            emp, mcc, prt = [], [], []
            for b in pool.tolist():
                h = H[b:b + 1]
                i = int(logits[b].argmax())
                lo, hi = (h - eps).squeeze(0), (h + eps).squeeze(0)
                e = pgd_pair((W1[i], W2[i], W3[i]),
                             (W1q[i], W2q[i], W3q[i]), h, eps)
                m = swiglu_diff_bound_box(W1[i], W2[i], W3[i],
                                          W1q[i], W2q[i], W3q[i], lo, hi)
                p = swiglu_pert_bound(W1[i], W2[i], W3[i],
                                      W1q[i], W2q[i], W3q[i], lo, hi)
                emp.append(e); mcc.append(float(m)); prt.append(p)
            me = sorted(emp)[len(emp) // 2]
            mm = sorted(mcc)[len(mcc) // 2]
            mp = sorted(prt)[len(prt) // 2]
            print(f"  {eps:>6.3f} {me:>9.4f} {mm:>11.4f} {mp:>11.4f} "
                  f"{mm/max(me,1e-9):>9.1f}x {mp/max(me,1e-9):>10.1f}x")

    print(f"\n{'='*72}")
    print("  pert-bound must be >= PGD (sound) and << McCormick (tight). If so,")
    print("  the cancellation-aware bound unlocks tight certified quantization")
    print("  over eps-balls -> certified robust quantization on SwiGLU MoE.")


if __name__ == "__main__":
    main()
