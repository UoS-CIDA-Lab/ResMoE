"""Experiment 57 — Certified memory compression vs robustness radius epsilon.

For a real MoE (OLMoE-1B-7B layer-0), how much can we compress while CERTIFYING
output deviation <= delta over an L2 eps-ball? Larger eps (wider guarantee) ->
looser bound -> more bits needed -> less compression. We tabulate the certified
per-expert budget (min avg bits, % of fp16 memory) over a grid of (eps, delta),
plus the *maximum* certified compression per eps.

Uses the tight exp-54 bound; budget = coarsest bits in {8,6,4,3,2} per expert
whose certified bound (max over the expert's routed inputs) <= delta.
Run: python3 experiments/57_compression_vs_epsilon.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
BITS = [8, 6, 4, 3, 2]
EPSS = [0.01, 0.05, 0.1, 0.2]
DELTAS = [0.005, 0.01, 0.05, 0.1]


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def silu_grad(z):
    s = torch.sigmoid(z)
    return s + z * s * (1 - s)


def affine_box(W, lo, hi):
    c = (lo + hi) / 2; r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_consts():
    z = torch.linspace(-30, 30, 1200001).requires_grad_(True)
    g1, = torch.autograd.grad(silu(z).sum(), z, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    return (g1.abs().max().item()*1.01, g2.abs().max().item()*1.01, g3.abs().max().item()*1.01)


def quantize_per_channel(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / scale).clamp(-qmax - 1, qmax) * scale


@torch.no_grad()
def bigspec(W):
    """Exact largest singular value (SVD) -- SOUND. Used for static W1,W2."""
    return torch.linalg.matrix_norm(W, ord=2).item()


def diffspec(d):
    """Sound, cheap upper bound on ||d||_2: sqrt(||d||_1 * ||d||_inf)
    (max col-abs-sum times max row-abs-sum). Loose but d is tiny (quant error),
    so the absolute slack in the eps^2 remainder is negligible. No SVD."""
    n1 = d.abs().sum(0).max().item()      # ||d||_1 (max column abs sum)
    ninf = d.abs().sum(1).max().item()    # ||d||_inf (max row abs sum)
    return (n1 * ninf) ** 0.5


@torch.no_grad()
def dh0_jd(W1, W2, W3, W1q, W2q, W3q, h0):
    """eps-INDEPENDENT (expensive, once per (expert,input,bit)):
    Dh0[o]=|E_o(h0)-E_q,o(h0)|, JDl2[o]=||J_D[o,:](h0)||_2 (exact Jacobian-diff)."""
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JDl2 = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    return Dh0, JDl2


@torch.no_grad()
def hd_vec(W2, W2q, W3, W3q, d1, d2, h0, eps, sv, Ls, Lpp, Lppp):
    """eps-dependent spectral-Hessian bound HD[o] (cheap: matvec + elementwise)."""
    s1, s2, s1q, s2q, sd1, sd2 = sv
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi)); Bvq = amax(*affine_box(W2q, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi)); Bd2 = amax(*affine_box(d2, lo, hi))
    d3 = W3 - W3q
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)
    De = Ls * W3.abs().amax(1)
    dDc = (W3.abs()*(Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0) + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs()*(Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    return Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)


def main():
    print("Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float(); H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    Ls, Lpp, Lppp = silu_consts()
    _, tk = (H @ Wg.T).topk(K, dim=-1)
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool); inset.scatter_(1, tk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist() for e in active}
    # cert[e][eps][b] — quantize per-expert on the fly (memory-light)
    print(f"Computing certified bounds over {len(active)} experts x "
          f"{len(EPSS)} eps x {len(BITS)} bits...", flush=True)
    cert = {e: {ep: {} for ep in EPSS} for e in active}
    for e in active:
        s1, s2 = bigspec(W1[e]), bigspec(W2[e])   # 2 SVDs per expert (static)
        inputs = routed[e][:2]                    # cap inputs (max over them)
        for b in BITS:
            W1q = quantize_per_channel(W1[e], b)
            W2q = quantize_per_channel(W2[e], b)
            W3q = quantize_per_channel(W3[e], b)
            d1, d2 = W1[e] - W1q, W2[e] - W2q
            # exact SVD (sound + tight, consistent with the RQ2 budget table);
            # affordable now that spectral norms are hoisted out of the eps/input loops
            sd1, sd2 = bigspec(d1), bigspec(d2)
            sv = (s1, s2, bigspec(W1q), bigspec(W2q), sd1, sd2)
            # eps-independent (expensive) per input, computed once:
            pre = [(t, *dh0_jd(W1[e], W2[e], W3[e], W1q, W2q, W3q, H[t:t+1]))
                   for t in inputs]
            for ep in EPSS:
                bnd = 0.0
                for t, Dh0, JDl2 in pre:
                    HD = hd_vec(W2[e], W2q, W3[e], W3q, d1, d2, H[t:t+1],
                                ep, sv, Ls, Lpp, Lppp)
                    bnd = max(bnd, (Dh0 + ep*JDl2 + 0.5*ep*ep*HD).max().item())
                cert[e][ep][b] = bnd
        print(f"  expert {e} done", flush=True)

    def budget(ep, dmax):
        avg = sum(min([b for b in BITS if cert[e][ep][b] <= dmax] or [16]) for e in active) / len(active)
        return avg

    print(f"\n{'='*64}")
    print(f"  Certified memory compression vs eps (OLMoE layer-0, {len(active)} experts)")
    print(f"  cells = avg bits  ({'%'} of fp16 memory);  16=forced fp16 (no compress)")
    print(f"{'='*64}")
    header = "  eps \\ delta " + "".join(f"{d:>14.3f}" for d in DELTAS)
    print(header)
    for ep in EPSS:
        row = f"  {ep:>10.2f}  "
        for d in DELTAS:
            avg = budget(ep, d)
            row += f"{avg:>6.2f}b/{avg/16*100:>4.0f}% "
        print(row)

    print(f"\n  MAX certified compression per eps (most aggressive, any delta<=0.1):")
    print(f"  {'eps':>8s} {'min avg bits':>13s} {'min mem vs fp16':>16s}")
    for ep in EPSS:
        best = min(budget(ep, d) for d in DELTAS)
        print(f"  {ep:>8.2f} {best:>13.2f} {best/16*100:>15.1f}%")
    print(f"\n  => larger eps (wider certified radius) costs compression: the")
    print("  robustness radius you certify against trades off against memory.")


if __name__ == "__main__":
    main()
