"""Experiment 54 — (b) Tighten the large-eps L2 bound via the DIFFERENCE-Hessian
spectral norm (dW-aware).

exp 52's remainder used (1/2)eps^2 (||H_E||_2 + ||H_Eq||_2) — the full Hessian
spectral of both experts (triangle bound on ||H_D||_2), which grows at large eps
(8-bit 23x@0.05, 104x@0.1). But D = E - E_q has Hessian exactly H_D = H_E - H_Eq,
which is ~dW (since W~Wq). We bound ||H_D||_2 directly with the difference
identity  A^T X A - Aq^T Xq Aq = A^T X dA + dA^T X Aq + Aq^T dX Aq  (dA=A-Aq).
With H_{E_o} = W1^T Dc W1 + (W1^T De W2 + W2^T De W1), Dc=diag(W3[o,k]s''(u)v),
De=diag(W3[o,k]s'(u)):
  ||H_{D,o}||_2 <= ||Dc||_2 ||d1||_2 (||W1||_2+||W1q||_2) + ||W1q||_2^2 ||Dc-Dcq||_2
   + 2[ ||W1||_2 ||De||_2 ||d2||_2 + ||d1||_2 ||De||_2 ||W2q||_2
        + ||W1q||_2 ||De-Deq||_2 ||W2q||_2 ]
where ||Dc||_2=max_k|...| (diagonal) and ||Dc-Dcq||_2, ||De-Deq||_2 ~ dW (bounded
via product-difference with s',s'',s''' Lipschitz consts). Each term carries a dW
factor (||d1||_2, ||d2||_2, ||Dc-Dcq||_2, ||De-Deq||_2).

Bound: |D_o(x)| <= |D_o(h0)| + eps||J_D(h0)[o]||_2 + (1/2)eps^2 ||H_{D,o}||_2.
Compares PGD vs exp52(triangle) vs exp54(diff-Hessian). Run: python3 ...54...py
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
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    return (g1.abs().max().item() * 1.01, g2.abs().max().item() * 1.01,
            g3.abs().max().item() * 1.01)


def quantize_per_channel(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / scale).clamp(-qmax - 1, qmax) * scale


def jac_diff_l2(W1, W2, W3, W1q, W2q, W3q, h0):
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C * (silu_grad(u) * v).unsqueeze(0)) @ A + (C * F.silu(u).unsqueeze(0)) @ B
    return (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)


def diffhess_spectral(W3, W3q, lo, hi, sp, Ls, Lpp, Lppp):
    """||H_{D,o}||_2 bound per output o (dW-aware), spectral."""
    s1, s2, s1q, s2q, sd1, sd2 = sp
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(_W2, lo, hi))
    Bvq = amax(*affine_box(_W2q, lo, hi))
    Bd1 = amax(*affine_box(_W1 - _W1q, lo, hi))
    Bd2 = amax(*affine_box(_W2 - _W2q, lo, hi))
    d3 = (W3 - W3q).abs()
    # diagonal spectral norms (max over k), per output o
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)            # ||Dc||_2
    De = Ls * W3.abs().amax(1)                                  # ||De||_2
    # ||Dc - Dcq||_2 = max_k |c_k - cq_k|, c~W3 s''(u) v
    dc_k = (W3.abs() * (Lpp * Bd2 + Lppp * Bd1 * Bvq).unsqueeze(0)
            + d3 * (Lpp * Bvq).unsqueeze(0))                    # [d_out, d_ff]
    dDc = dc_k.amax(1)
    de_k = W3.abs() * (Lpp * Bd1).unsqueeze(0) + d3 * Ls        # |e_k - eq_k|
    dDe = de_k.amax(1)
    block_c = Dc * sd1 * (s1 + s1q) + s1q * s1q * dDc
    block_e = 2 * (s1 * De * sd2 + sd1 * De * s2q + s1q * dDe * s2q)
    return block_c + block_e                                    # [d_out]


def bound54(W1, W2, W3, W1q, W2q, W3q, h0, eps, sp, Ls, Lpp, Lppp):
    global _W1, _W2, _W1q, _W2q
    _W1, _W2, _W1q, _W2q = W1, W2, W1q, W2q
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    HD = diffhess_spectral(W3, W3q, lo, hi, sp, Ls, Lpp, Lppp)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    JD = jac_diff_l2(W1, W2, W3, W1q, W2q, W3q, h0)
    return (Dh0 + eps * JD + 0.5 * eps * eps * HD).max().item()


def bound52(W1, W2, W3, W1q, W2q, W3q, h0, eps, sp, Ls, Lpp):
    """exp 52: triangle ||H_E||_2 + ||H_Eq||_2."""
    s1, s2, s1q, s2q, _, _ = sp
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi))
    Bvq = amax(*affine_box(W2q, lo, hi))
    HE = s1 * s1 * Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1) + 2 * s1 * s2 * Ls * W3.abs().amax(1)
    HEq = s1q * s1q * Lpp * (W3q.abs() * Bvq.unsqueeze(0)).amax(1) + 2 * s1q * s2q * Ls * W3q.abs().amax(1)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    JD = jac_diff_l2(W1, W2, W3, W1q, W2q, W3q, h0)
    return (Dh0 + eps * JD + 0.5 * eps * eps * (HE + HEq)).max().item()


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
            h = h + step * g / g.norm().clamp(min=1e-12)
            delta = h - h0
            dn = delta.norm()
            if dn > eps:
                h = h0 + delta * (eps / dn)
            best = max(best, (swiglu(h, a, b, d) - swiglu(h, aq, bq, dq)
                              ).abs().max(-1).values.max().item())
    return best


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
        sn = lambda W: torch.linalg.matrix_norm(W, ord=2).item()
        SP = {i: (sn(W1[i]), sn(W2[i]), sn(W1q[i]), sn(W2q[i]),
                  sn(W1[i] - W1q[i]), sn(W2[i] - W2q[i])) for i in range(N)}
        print(f"\n{'='*70}\n  {bits}-bit: L2-PGD vs exp52(triangle) vs exp54(diff-Hess)"
              f"\n{'='*70}")
        print(f"  {'eps':>6s} {'L2-PGD':>10s} {'exp52':>10s} {'exp54':>10s} "
              f"{'52×':>6s} {'54×':>6s} {'sound?':>7s}")
        print("  " + "-" * 58)
        for eps in (0.0, 0.01, 0.05, 0.1, 0.2):
            E, B52, B54 = [], [], []
            for bi in pool.tolist():
                h = H[bi:bi + 1]
                i = int(logits[bi].argmax())
                Wi = (W1[i], W2[i], W3[i]); Wj = (W1q[i], W2q[i], W3q[i])
                E.append(l2_pgd(Wi, Wj, h, eps))
                B52.append(bound52(*Wi, *Wj, h, eps, SP[i], Ls, Lpp))
                B54.append(bound54(*Wi, *Wj, h, eps, SP[i], Ls, Lpp, Lppp))
            me = sorted(E)[len(E) // 2]
            m52 = sorted(B52)[len(B52) // 2]; m54 = sorted(B54)[len(B54) // 2]
            sound = "yes" if all(b >= e - 1e-6 for b, e in zip(B54, E)) else "NO!"
            print(f"  {eps:>6.3f} {me:>10.4f} {m52:>10.4f} {m54:>10.4f} "
                  f"{m52/max(me,1e-9):>5.0f}× {m54/max(me,1e-9):>5.1f}× {sound:>7s}")

    print(f"\n{'='*70}")
    print("  exp54 (diff-Hessian, ~dW) should stay tight at large eps where exp52")
    print("  (full Hessian) blew up. Must be sound (>= PGD).")


if __name__ == "__main__":
    main()
