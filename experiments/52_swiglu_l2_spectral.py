"""Experiment 52 — L2-ball certified quant robustness via SPECTRAL Hessian.

The L_inf wall (exp 50/51): over an L_inf ball, delta^T H delta has rank-1 terms
(w.delta)^2 with worst case ||w||_1^2 eps^2 — a 2048-dim L1-SQUARED inflation no
entrywise bound avoids. Escape: certify over an L2 ball. Then
  max_{||delta||_2<=eps} |delta^T H delta| = eps^2 ||H||_2  (spectral norm),
and the rank-1 term w⊗w has spectral norm ||w||_2^2 (NOT ||w||_1^2), with a MAX
over k (not a sum). So even the FULL Hessian spectral bound avoids both
inflations:
  ||H_{E_o}||_2 <= ||W1||_2^2 * max_k|W3[o,k] sigma''(u_k) v_k|
                + 2 ||W1||_2 ||W2||_2 * max_k|W3[o,k] sigma'(u_k)|.
Bound (Taylor, exact first-order in L2):
  |D_o(x)| <= |D_o(h0)| + eps*||J_D(h0)[o,:]||_2
            + (1/2) eps^2 (||H_{E_o}||_2 + ||H_{Eq_o}||_2).
||W||_2 = largest singular value (precomputed per expert).

Compares L2-PGD vs this spectral bound. Must be SOUND (>= L2-PGD). If ~1-3x,
SwiGLU certified-quant robustness is SOLVED (in the L2 threat model).
Run: python3 experiments/52_swiglu_l2_spectral.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu

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
    g2, = torch.autograd.grad(g1.sum(), z)
    return g1.abs().max().item() * 1.01, g2.abs().max().item() * 1.01


def jac_diff_l2(W1, W2, W3, W1q, W2q, W3q, h0):
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C * (silu_grad(u) * v).unsqueeze(0)) @ A + (C * F.silu(u).unsqueeze(0)) @ B
    return (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)   # L2 per output row


def hess_spec_bound(W3, s1, s2, Bv, Ls, Lpp):
    """||H_{E_o}||_2 bound per output o (spectral)."""
    maxc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)      # max_k |W3 sigma'' v|
    maxe = Ls * W3.abs().amax(1)                            # max_k |W3 sigma'|
    return s1 * s1 * maxc + 2 * s1 * s2 * maxe              # [d_out]


def l2_bound(W1, W2, W3, W1q, W2q, W3q, h0, eps, Ls, Lpp, sp):
    s1, s2, s1q, s2q = sp
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps   # superset of L2 ball
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi))
    Bvq = amax(*affine_box(W2q, lo, hi))
    HE = hess_spec_bound(W3, s1, s2, Bv, Ls, Lpp)
    HEq = hess_spec_bound(W3q, s1q, s2q, Bvq, Ls, Lpp)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    JD2 = jac_diff_l2(W1, W2, W3, W1q, W2q, W3q, h0)
    return (Dh0 + eps * JD2 + 0.5 * eps * eps * (HE + HEq)).max().item()


def l2_pgd(Wi, Wj, h0, eps, n_steps=200):
    (a, b, d), (aq, bq, dq) = Wi, Wj
    step = eps / 20 if eps > 0 else 0.0
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps if eps > 0 else 1):
        hg = h.clone().requires_grad_(True)
        loss = (swiglu(hg, a, b, d) - swiglu(hg, aq, bq, dq)).abs().max(-1).values.sum()
        loss.backward()
        with torch.no_grad():
            g = hg.grad
            gn = g.norm().clamp(min=1e-12)
            h = h + step * g / gn                      # L2 steepest ascent
            delta = h - h0
            dn = delta.norm()
            if dn > eps:
                h = h0 + delta * (eps / dn)            # project to L2 ball
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
    Ls, Lpp = silu_consts()
    print(f"  SiLU: sup|s'|={Ls:.3f}, sup|s''|={Lpp:.3f}")
    torch.manual_seed(0)
    pool = torch.randperm(H.shape[0])[:20]

    for bits in (8, 4):
        W1q = torch.stack([quantize_per_channel(W1[i], bits) for i in range(N)])
        W2q = torch.stack([quantize_per_channel(W2[i], bits) for i in range(N)])
        W3q = torch.stack([quantize_per_channel(W3[i], bits) for i in range(N)])
        # precompute spectral norms per expert
        sn = lambda W: torch.linalg.matrix_norm(W, ord=2)
        s1 = [sn(W1[i]).item() for i in range(N)]
        s2 = [sn(W2[i]).item() for i in range(N)]
        s1q = [sn(W1q[i]).item() for i in range(N)]
        s2q = [sn(W2q[i]).item() for i in range(N)]
        print(f"\n{'='*66}\n  {bits}-bit quant: L2-PGD vs L2-SPECTRAL bound(exp52)"
              f"\n{'='*66}")
        print(f"  {'eps':>6s} {'L2-PGD':>10s} {'L2-bound':>11s} {'bound×':>8s} {'sound?':>7s}")
        print("  " + "-" * 46)
        for eps in (0.0, 0.01, 0.05, 0.1):
            E, B = [], []
            for bi in pool.tolist():
                h = H[bi:bi + 1]
                i = int(logits[bi].argmax())
                Wi = (W1[i], W2[i], W3[i]); Wj = (W1q[i], W2q[i], W3q[i])
                E.append(l2_pgd(Wi, Wj, h, eps))
                B.append(l2_bound(*Wi, *Wj, h, eps, Ls, Lpp,
                                  (s1[i], s2[i], s1q[i], s2q[i])))
            me = sorted(E)[len(E) // 2]; mb = sorted(B)[len(B) // 2]
            sound = "yes" if all(bb >= ee - 1e-6 for bb, ee in zip(B, E)) else "NO!"
            print(f"  {eps:>6.3f} {me:>10.4f} {mb:>11.4f} {mb/max(me,1e-9):>7.1f}× {sound:>7s}")

    print(f"\n{'='*66}")
    print("  bound× ~ 1-3x AND sound => SwiGLU certified-quant robustness SOLVED")
    print("  in the L2 threat model. Spectral norm dodges the L_inf ||w||_1^2 wall.")


if __name__ == "__main__":
    main()
