"""Experiment 49 — Jacobian-difference bound for SwiGLU quantization (extension A).

exp 48's diagnosis: the empirical quant error is nearly CONSTANT over the
eps-ball (gradient of E - E_q is small ~ dW), but prior bounds add eps-growing
slack proportional to a FULL expert magnitude. Fix: bound the difference by its
value at the center (exact) plus eps times the row-L1 of the JACOBIAN DIFFERENCE
J_D = J_E - J_Eq, which is itself ~ dW:
    |D_o(x)| <= |D_o(h0)| + eps * max_xi ||J_D[o,:](xi)||_1            (MVT)

Jacobian of SwiGLU: with u=W1 x, v=W2 x, J_E[o,m] = sum_k W3[o,k] Jg_k[m],
  Jg_k[m] = sigma'(u_k) W1[k,m] v_k + sigma(u_k) W2[k,m].
J_D[o,:] = sum_k W3[o,k](Jg_k - Jg'_k) + sum_k d3[o,k] Jg'_k, and (expanding to
expose the dW factors d1=W1-W1q etc.):
  ||Jg_k - Jg'_k||_1 <= |p_k| ||d1[k]||_1 + |p_k - p'_k| ||W1q[k]||_1
                      + |sigma(u_k)| ||d2[k]||_1 + |sigma(u_k)-sigma(u'_k)| ||W2q[k]||_1
  with p=sigma'(u)v: |p|<=Lsig*Bv, |p-p'|<=Lsig*Bd2 + Lsigpp*Bd1*Bvq,
  |sigma(u)-sigma(u')|<=Lsig*Bd1.
  ||Jg'_k||_1 <= Lsig*Bvq*||W1q[k]||_1 + |sigma(u'_k)|*||W2q[k]||_1.
The dW-bearing terms (d1,d2,Bd1,Bd2,p-p') make the eps-term scale ~ eps*dW.

Compares PGD vs McCormick vs signed(exp48) vs THIS jacdiff bound. SOUND check:
must stay >= PGD. Run: python3 experiments/49_swiglu_jacdiff_bound.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import swiglu_diff_bound_box, silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
SILU_MIN_Z, SILU_MIN_V = -1.2784, -0.2785


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def affine_box(W, lo, hi):
    c = (lo + hi) / 2
    r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_absmax(a, b):
    m = torch.maximum(silu(a).abs(), silu(b).abs())
    straddle = (a <= SILU_MIN_Z) & (SILU_MIN_Z <= b)
    return torch.where(straddle, torch.maximum(m, torch.full_like(m, abs(SILU_MIN_V))), m)


def silu_lipschitz_consts():
    """Sound global bounds on sup|sigma'| and sup|sigma''| (SiLU)."""
    z = torch.linspace(-30, 30, 600001)
    z.requires_grad_(True)
    s = silu(z)
    g, = torch.autograd.grad(s.sum(), z, create_graph=True)
    gg, = torch.autograd.grad(g.sum(), z)
    return g.abs().max().item() * 1.01, gg.abs().max().item() * 1.01


def quantize_per_channel(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / scale).clamp(-qmax - 1, qmax) * scale


def jacdiff_bound(W1, W2, W3, W1q, W2q, W3q, h0, eps, Lsig, Lsigpp):
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    c = (lo + hi) / 2
    r = (hi - lo) / 2
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    # ball magnitudes (per hidden unit k)
    v_lo, v_hi = affine_box(W2, lo, hi)
    vq_lo, vq_hi = affine_box(W2q, lo, hi)
    u_lo, u_hi = affine_box(W1, lo, hi)
    uq_lo, uq_hi = affine_box(W1q, lo, hi)
    Bv = amax(v_lo, v_hi)
    Bvq = amax(vq_lo, vq_hi)
    Bd1 = (d1 @ c).abs() + d1.abs() @ r
    Bd2 = (d2 @ c).abs() + d2.abs() @ r
    Su = silu_absmax(u_lo, u_hi)
    Suq = silu_absmax(uq_lo, uq_hi)
    # fixed row-L1 norms
    nd1, nd2 = d1.abs().sum(1), d2.abs().sum(1)
    nW1q, nW2q = W1q.abs().sum(1), W2q.abs().sum(1)
    # per-k ||Jg_k - Jg'_k||_1  (all terms carry a dW factor)
    p_abs = Lsig * Bv                                  # |sigma'(u) v|
    pp_diff = Lsig * Bd2 + Lsigpp * Bd1 * Bvq          # |p - p'|
    Ljg_diff = (p_abs * nd1 + pp_diff * nW1q
                + Su * nd2 + (Lsig * Bd1) * nW2q)      # [d_ff]
    # per-k ||Jg'_k||_1  (full magnitude, multiplied by tiny d3)
    Ljgq = Lsig * Bvq * nW1q + Suq * nW2q              # [d_ff]
    # row-L1 of J_D per output coord
    L_D = W3.abs() @ Ljg_diff + d3.abs() @ Ljgq        # [d_out]
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps * L_D).max().item()


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


def main():
    print("Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float()
    H = c["H"].float()
    N = c["n_experts"]
    logits = H @ Wg.T
    Lsig, Lsigpp = silu_lipschitz_consts()
    print(f"  SiLU bounds: sup|sigma'|={Lsig:.3f}, sup|sigma''|={Lsigpp:.3f}")
    torch.manual_seed(0)
    pool = torch.randperm(H.shape[0])[:20]

    for bits in (8, 4):
        W1q = torch.stack([quantize_per_channel(W1[i], bits) for i in range(N)])
        W2q = torch.stack([quantize_per_channel(W2[i], bits) for i in range(N)])
        W3q = torch.stack([quantize_per_channel(W3[i], bits) for i in range(N)])
        print(f"\n{'='*72}\n  {bits}-bit quant: PGD vs McCormick vs JACDIFF(exp49)"
              f"\n{'='*72}")
        print(f"  {'eps':>6s} {'PGD':>10s} {'McCormick':>11s} {'jacdiff':>11s} "
              f"{'McC×':>8s} {'jac×':>7s} {'sound?':>7s}")
        print("  " + "-" * 62)
        for eps in (0.0, 0.01, 0.05, 0.1):
            E, M, J = [], [], []
            for b in pool.tolist():
                h = H[b:b + 1]
                i = int(logits[b].argmax())
                Wi = (W1[i], W2[i], W3[i]); Wj = (W1q[i], W2q[i], W3q[i])
                E.append(pgd_pair(Wi, Wj, h, eps))
                M.append(swiglu_diff_bound_box(*Wi, *Wj, (h - eps).squeeze(0), (h + eps).squeeze(0)))
                J.append(jacdiff_bound(*Wi, *Wj, h, eps, Lsig, Lsigpp))
            me = sorted(E)[len(E) // 2]; mm = sorted(M)[len(M) // 2]
            mj = sorted(J)[len(J) // 2]
            sound = "yes" if all(j >= e - 1e-6 for j, e in zip(J, E)) else "NO!"
            print(f"  {eps:>6.3f} {me:>10.4f} {mm:>11.3f} {mj:>11.4f} "
                  f"{mm/max(me,1e-9):>7.0f}× {mj/max(me,1e-9):>6.1f}× {sound:>7s}")

    print(f"\n{'='*72}")
    print("  jac× should be ~O(1-10)x (eps-term now scales eps*dW). Must be sound")
    print("  (jacdiff >= PGD for every sample). If so: tight certified quant")
    print("  robustness on SwiGLU MoE — the open problem is solved.")


if __name__ == "__main__":
    main()
