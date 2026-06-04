"""Experiment 51 — Taylor bound with a DIFFERENCE-aware 2nd-order remainder.

Diagnostic (exp 50b) confirmed: D(h0) + eps*||J_D(h0)||_1 matches empirical at
1.1-1.8x (the exact first-order Jacobian-difference is tight). The ONLY missing
piece for a SOUND bound is the 2nd-order remainder, which exp 50 bounded by the
FULL Hessian (||W1||_1^2) -> exploded. Fix: bound the remainder by the
DIFFERENCE Hessian H_D = H_E - H_Eq, every term carrying a dW factor via
product-difference identities, e.g. (W1.d)^2-(W1q.d)^2 = (dW1.d)((W1+W1q).d):

  |delta^T H_{D,o} delta| <= eps^2 [ sum_k |W3[o,k]| dPhi_k + sum_k |d3[o,k]| Qq_k ]
  dPhi_k (all terms ~ dW):
    Term1 = Lspp*Bv*nd1*(nW1+nW1q) + (Lspp*Bd2 + Lsppp*Bd1*Bvq)*nW1q^2
    Term2 = 2[ Lsig*(nW1*nd2 + nd1*nW2q) + Lspp*Bd1*nW1q*nW2q ]
  Qq_k (full, x tiny d3) = Lspp*Bvq*nW1q^2 + 2*Lsig*nW1q*nW2q
where n*=row L1 norms, B*=max|.| over the ball, Bd1/Bd2 = max|dW.x| over ball.

  |D_o(x)| <= |D_o(h0)| + eps*||J_D(h0)[o,:]||_1 + (1/2) eps^2 * R_o
Must be SOUND (>= PGD) and ideally ~2-5x. Run: python3 experiments/51_...py
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
    z = torch.linspace(-30, 30, 1200001).requires_grad_(True)
    g1, = torch.autograd.grad(silu(z).sum(), z, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    return (g1.abs().max().item() * 1.01, g2.abs().max().item() * 1.01,
            g3.abs().max().item() * 1.01)


def jac_diff_l1(W1, W2, W3, W1q, W2q, W3q, h0):
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C * (silu_grad(u) * v).unsqueeze(0)) @ A + (C * F.silu(u).unsqueeze(0)) @ B
    return (J(W1, W2, W3) - J(W1q, W2q, W3q)).abs().sum(1)


def taylor_diffhess(W1, W2, W3, W1q, W2q, W3q, h0, eps, Ls, Lpp, Lppp):
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi))
    Bvq = amax(*affine_box(W2q, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi))
    Bd2 = amax(*affine_box(d2, lo, hi))
    nW1, nW2 = W1.abs().sum(1), W2.abs().sum(1)
    nW1q, nW2q = W1q.abs().sum(1), W2q.abs().sum(1)
    nd1, nd2 = d1.abs().sum(1), d2.abs().sum(1)
    # difference-Hessian per-k quadratic-form magnitude (unit delta), all ~dW
    Term1 = Lpp * Bv * nd1 * (nW1 + nW1q) + (Lpp * Bd2 + Lppp * Bd1 * Bvq) * nW1q ** 2
    Term2 = 2 * (Ls * (nW1 * nd2 + nd1 * nW2q) + Lpp * Bd1 * nW1q * nW2q)
    dPhi = Term1 + Term2                                  # [d_ff]
    Qq = Lpp * Bvq * nW1q ** 2 + 2 * Ls * nW1q * nW2q     # full, x |d3|
    R = W3.abs() @ dPhi + d3.abs() @ Qq                   # [d_out]
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    JD1 = jac_diff_l1(W1, W2, W3, W1q, W2q, W3q, h0)
    return (Dh0 + eps * JD1 + 0.5 * eps * eps * R).max().item()


def pgd_pair(Wi, Wj, h0, eps, n_steps=150):
    (a, b, d), (aq, bq, dq) = Wi, Wj
    step = eps / 15 if eps > 0 else 0.0
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps if eps > 0 else 1):
        hg = h.clone().requires_grad_(True)
        l = (swiglu(hg, a, b, d) - swiglu(hg, aq, bq, dq)).abs().max(-1).values.sum()
        l.backward()
        with torch.no_grad():
            if eps > 0:
                h = torch.max(torch.min(h + step * hg.grad.sign(), h0 + eps), h0 - eps)
            best = max(best, (swiglu(h, a, b, d) - swiglu(h, aq, bq, dq)
                              ).abs().max(-1).values.max().item())
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
    Ls, Lpp, Lppp = silu_consts()
    print(f"  SiLU: sup|s'|={Ls:.3f}, sup|s''|={Lpp:.3f}, sup|s'''|={Lppp:.3f}")
    torch.manual_seed(0)
    pool = torch.randperm(H.shape[0])[:20]

    for bits in (8, 4):
        W1q = torch.stack([quantize_per_channel(W1[i], bits) for i in range(N)])
        W2q = torch.stack([quantize_per_channel(W2[i], bits) for i in range(N)])
        W3q = torch.stack([quantize_per_channel(W3[i], bits) for i in range(N)])
        print(f"\n{'='*72}\n  {bits}-bit quant: PGD vs McCormick vs TAYLOR+diffHess(exp51)"
              f"\n{'='*72}")
        print(f"  {'eps':>6s} {'PGD':>10s} {'McCormick':>11s} {'taylor51':>11s} "
              f"{'McC×':>8s} {'t51×':>7s} {'sound?':>7s}")
        print("  " + "-" * 62)
        for eps in (0.0, 0.01, 0.05, 0.1):
            E, M, T = [], [], []
            for b in pool.tolist():
                h = H[b:b + 1]
                i = int(logits[b].argmax())
                Wi = (W1[i], W2[i], W3[i]); Wj = (W1q[i], W2q[i], W3q[i])
                E.append(pgd_pair(Wi, Wj, h, eps))
                M.append(swiglu_diff_bound_box(*Wi, *Wj, (h - eps).squeeze(0), (h + eps).squeeze(0)))
                T.append(taylor_diffhess(*Wi, *Wj, h, eps, Ls, Lpp, Lppp))
            me = sorted(E)[len(E) // 2]; mm = sorted(M)[len(M) // 2]
            mt = sorted(T)[len(T) // 2]
            sound = "yes" if all(t >= e - 1e-6 for t, e in zip(T, E)) else "NO!"
            print(f"  {eps:>6.3f} {me:>10.4f} {mm:>11.3f} {mt:>11.4f} "
                  f"{mm/max(me,1e-9):>7.0f}× {mt/max(me,1e-9):>6.1f}× {sound:>7s}")

    print(f"\n{'='*72}")
    print("  t51× ~ 2-5x AND sound => SwiGLU certified-quant robustness SOLVED.")


if __name__ == "__main__":
    main()
