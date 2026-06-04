"""Experiment 50 — Taylor bound: EXACT first-order Jacobian-diff + 2nd-order
remainder. The technique that should crack the SwiGLU adversarial wall.

exp 49 tried to BOUND ||J_D||_1 over the ball and lost the k-sum cancellation in
the triangle inequality. The fix: don't bound the Jacobian — COMPUTE the
Jacobian difference J_D(h0) EXACTLY at the center (analytically), so all k-sum
and m cancellation is captured numerically, then bound only the SECOND-order
remainder (which is eps^2 * Hessian-diff ~ eps^2 * dW, doubly small):
    |D_o(x)| <= |D_o(h0)| + eps*||J_D(h0)[o,:]||_1 + (1/2) eps^2 * R_o
J_E(x) = (W3 * (sigma'(u)*v)) @ W1 + (W3 * sigma(u)) @ W2,  u=W1 x, v=W2 x  (exact).
R_o (sound Hessian-magnitude bound over the ball), with Q_k = sup|sigma''| |v_k|
||W1[k]||_1^2 + 2 sup|sigma'| ||W1[k]||_1 ||W2[k]||_1:
    R_o = sum_k |W3[o,k]|(Q_k + Q'_k) + sum_k |d3[o,k]| Q'_k.

Compares PGD vs McCormick vs Taylor. Must be SOUND (>= PGD). If Taylor× ~ 1-3x,
the SwiGLU certified-quant-robustness problem is solved.
Run: python3 experiments/50_swiglu_taylor_bound.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import swiglu_diff_bound_box, silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def silu_grad(z):
    s = torch.sigmoid(z)
    return s + z * s * (1 - s)


def affine_box(W, lo, hi):
    c = (lo + hi) / 2
    r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_consts():
    z = torch.linspace(-30, 30, 600001).requires_grad_(True)
    g, = torch.autograd.grad(silu(z).sum(), z, create_graph=True)
    gg, = torch.autograd.grad(g.sum(), z)
    return g.abs().max().item() * 1.01, gg.abs().max().item() * 1.01


def jacobian_diff_center(W1, W2, W3, W1q, W2q, W3q, h0):
    """Exact ||J_E - J_Eq||_1 row-wise at the point h0 (captures all cancellation)."""
    h = h0.squeeze(0)
    def jac(Wa, Wb, Wc):
        u = Wa @ h
        v = Wb @ h
        sg = silu_grad(u)
        s = F.silu(u)
        return (Wc * (sg * v).unsqueeze(0)) @ Wa + (Wc * s.unsqueeze(0)) @ Wb
    JD = jac(W1, W2, W3) - jac(W1q, W2q, W3q)        # [d_out, d]
    return JD.abs().sum(1)                            # [d_out]


def hessian_mag_bound(W1, W2, W3, lo, hi, Lsig, Lsigpp):
    """Sound bound on sum_{m,m'} |H_{E_o}[m,m']| per output o, over the box:
    Q_k = sup|sigma''| |v_k| ||W1[k]||_1^2 + 2 sup|sigma'| ||W1[k]||_1 ||W2[k]||_1."""
    v_lo, v_hi = affine_box(W2, lo, hi)
    Bv = torch.maximum(v_lo.abs(), v_hi.abs())
    nW1 = W1.abs().sum(1)
    nW2 = W2.abs().sum(1)
    Q = Lsigpp * Bv * nW1 ** 2 + 2 * Lsig * nW1 * nW2   # [d_ff]
    return Q


def taylor_bound(W1, W2, W3, W1q, W2q, W3q, h0, eps, Lsig, Lsigpp):
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    JD1 = jacobian_diff_center(W1, W2, W3, W1q, W2q, W3q, h0)
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    Q = hessian_mag_bound(W1, W2, W3, lo, hi, Lsig, Lsigpp)
    Qq = hessian_mag_bound(W1q, W2q, W3q, lo, hi, Lsig, Lsigpp)
    d3 = (W3 - W3q).abs()
    R = W3.abs() @ (Q + Qq) + d3 @ Qq                 # [d_out]
    return (Dh0 + eps * JD1 + 0.5 * eps * eps * R).max().item()


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
    Lsig, Lsigpp = silu_consts()
    print(f"  SiLU: sup|sigma'|={Lsig:.3f}, sup|sigma''|={Lsigpp:.3f}")
    torch.manual_seed(0)
    pool = torch.randperm(H.shape[0])[:20]

    for bits in (8, 4):
        W1q = torch.stack([quantize_per_channel(W1[i], bits) for i in range(N)])
        W2q = torch.stack([quantize_per_channel(W2[i], bits) for i in range(N)])
        W3q = torch.stack([quantize_per_channel(W3[i], bits) for i in range(N)])
        print(f"\n{'='*74}\n  {bits}-bit quant: PGD vs McCormick vs TAYLOR(exp50)"
              f"\n{'='*74}")
        print(f"  {'eps':>6s} {'PGD':>10s} {'McCormick':>11s} {'taylor':>11s} "
              f"{'McC×':>8s} {'tay×':>7s} {'sound?':>7s}")
        print("  " + "-" * 62)
        for eps in (0.0, 0.01, 0.05, 0.1):
            E, M, T = [], [], []
            for b in pool.tolist():
                h = H[b:b + 1]
                i = int(logits[b].argmax())
                Wi = (W1[i], W2[i], W3[i]); Wj = (W1q[i], W2q[i], W3q[i])
                E.append(pgd_pair(Wi, Wj, h, eps))
                M.append(swiglu_diff_bound_box(*Wi, *Wj, (h - eps).squeeze(0), (h + eps).squeeze(0)))
                T.append(taylor_bound(*Wi, *Wj, h, eps, Lsig, Lsigpp))
            me = sorted(E)[len(E) // 2]; mm = sorted(M)[len(M) // 2]
            mt = sorted(T)[len(T) // 2]
            sound = "yes" if all(t >= e - 1e-6 for t, e in zip(T, E)) else "NO!"
            print(f"  {eps:>6.3f} {me:>10.4f} {mm:>11.3f} {mt:>11.4f} "
                  f"{mm/max(me,1e-9):>7.0f}× {mt/max(me,1e-9):>6.1f}× {sound:>7s}")

    print(f"\n{'='*74}")
    print("  tay× ~ 1-3x AND sound => SwiGLU certified-quant robustness SOLVED:")
    print("  exact first-order Jacobian-diff captures the k-sum cancellation,")
    print("  only the eps^2 * (Hessian ~ full) remainder is bounded loosely but")
    print("  is negligible for small eps.")


if __name__ == "__main__":
    main()
