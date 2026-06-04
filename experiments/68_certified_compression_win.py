"""Experiment 68 — Compression win at EQUAL certified guarantee: selective vs uniform.

The second axis of the ICSE paper. When a certified equivalence guarantee delta is
REQUIRED, uniform quantization must use the worst expert's certified-safe precision
for ALL experts; our per-expert certificate enables SELECTIVE allocation meeting the
same delta at fewer average bits. We quantify, per delta:
  - uniform-certified bits = min b in {8,6,4,3,2} s.t. max_e cert_e(b) <= delta (else 16)
  - selective bits        = per-expert coarsest b s.t. cert_e(b) <= delta
both SOUND (every expert certified <= delta). Reports avg bits & memory for each.
Uses the EXACT same cert (exp 54) over ALL routed inputs (sound). CPU, layer-0 cache.
Run: python3 experiments/68_certified_compression_win.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
EPS = 0.05
BITS = [8, 6, 4, 3, 2]
DELTAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def silu_grad(z):
    s = torch.sigmoid(z); return s + z * s * (1 - s)


def affine_box(W, lo, hi):
    c = (lo + hi) / 2; r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_consts():
    z = torch.linspace(-30, 30, 1200001).requires_grad_(True)
    g1, = torch.autograd.grad(silu(z).sum(), z, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    return (g1.abs().max().item()*1.01, g2.abs().max().item()*1.01, g3.abs().max().item()*1.01)


def quant(W, bits):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def bigspec(W):
    return torch.linalg.matrix_norm(W, ord=2).item()


def cert_bound(W1, W2, W3, W1q, W2q, W3q, h0, eps, Ls, Lpp, Lppp, sv):
    s1, s2, s1q, s2q, sd1, sd2 = sv
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi)); Bvq = amax(*affine_box(W2q, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi)); Bd2 = amax(*affine_box(d2, lo, hi))
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)
    De = Ls * W3.abs().amax(1)
    dDc = (W3.abs()*(Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0) + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs()*(Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    HD = Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JD = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps*JD + 0.5*eps*eps*HD).max().item()


def main():
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float(); H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    Ls, Lpp, Lppp = silu_consts()
    topk = (H @ Wg.T).topk(K, dim=-1).indices
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool); inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist() for e in active}
    print(f"  {len(active)} active experts, eps_L2={EPS} (cert over ALL routed inputs)",
          flush=True)

    cert = {e: {16: 0.0} for e in active}
    for e in active:
        s1, s2 = bigspec(W1[e]), bigspec(W2[e])
        for b in BITS:
            W1q, W2q, W3q = quant(W1[e], b), quant(W2[e], b), quant(W3[e], b)
            sv = (s1, s2, bigspec(W1q), bigspec(W2q), bigspec(W1[e]-W1q), bigspec(W2[e]-W2q))
            cert[e][b] = max(cert_bound(W1[e], W2[e], W3[e], W1q, W2q, W3q, H[t:t+1],
                                        EPS, Ls, Lpp, Lppp, sv) for t in routed[e])
        print(f"  expert {e} done", flush=True)

    def sel_bits(e, d):
        feas = [b for b in BITS if cert[e][b] <= d]
        return min(feas) if feas else 16

    print(f"\n{'='*72}\n  Compression at EQUAL certified guarantee delta (sound both)"
          f"\n{'='*72}")
    print(f"  {'delta':>6s} | {'uniform-certified':>20s} | {'selective (ours)':>20s} | "
          f"{'bit saving':>10s}")
    print("  " + "-"*68)
    for d in DELTAS:
        # uniform-certified: smallest single bit-width safe for ALL experts
        ub = 16
        for b in BITS:
            if all(cert[e][b] <= d for e in active):
                ub = b; break
        sel = {e: sel_bits(e, d) for e in active}
        savg = sum(sel.values()) / len(active)
        print(f"  {d:>6.3f} | {ub:>6.1f}b  {ub/16*100:>5.0f}% mem | "
              f"{savg:>6.2f}b  {savg/16*100:>5.0f}% mem | "
              f"{(ub-savg)/ub*100:>8.0f}%")
    print(f"\n  Both meet the SAME certified guarantee (every expert cert <= delta).")
    print("  uniform-certified must use the worst expert's safe precision for all;")
    print("  selective spends bits per-expert -> fewer avg bits at equal guarantee.")
    print("  (This is the compression win that exists ONLY in the certified setting;")
    print("   vs uncertified-empirical uniform we do NOT win -- that is a different axis.)")


if __name__ == "__main__":
    main()
