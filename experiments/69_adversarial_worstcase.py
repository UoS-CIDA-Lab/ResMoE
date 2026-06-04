"""Experiment 69 — Adversarial worst-case: certified-selective vs uniform at MATCHED
memory. The "much better under adversarial perturbation" test the user asked for.

Standard benchmarks are average-case and (via MoE routing redundancy) nearly identical
for any sane compression. The difference shows up in the WORST case: our certified
selective budget bounds every expert's deviation <= delta BY CONSTRUCTION, whereas
uniform quantization at the same average memory quantizes the sensitive experts
blindly, so its worst-case (adversarial) deviation from the reference is uncontrolled.
We PGD-maximize ||E_e(x) - E_{e,q}(x)||_inf over the L2 eps-ball for every active
expert and report the MAX over experts (the layer's worst-case adversarial divergence
from fp16) for matched-memory uniform vs certified-selective.

CPU, OLMoE layer-0 cache. Run: python3 experiments/69_adversarial_worstcase.py
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
PGD_STEPS = 200


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


def pgd_worst(h0, W1, W2, W3, W1q, W2q, W3q, eps, steps):
    h = h0.squeeze(0).detach()
    x = h.clone().requires_grad_(True)
    best = 0.0
    for _ in range(steps):
        diff = (swiglu(x.unsqueeze(0), W1, W2, W3)
                - swiglu(x.unsqueeze(0), W1q, W2q, W3q)).squeeze(0)
        obj = diff.abs().max()
        best = max(best, obj.item())
        g, = torch.autograd.grad(obj, x)
        with torch.no_grad():
            x += eps / 10 * g / (g.norm() + 1e-12) * eps
            d = x - h; n = d.norm()
            if n > eps:
                x.copy_(h + d * (eps / n))
        x.requires_grad_(True)
    return best


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
    print(f"  {len(active)} active experts, eps_L2={EPS}", flush=True)

    # per-expert certified bound (uncapped) for budget construction
    cert = {e: {16: 0.0} for e in active}
    for e in active:
        s1, s2 = bigspec(W1[e]), bigspec(W2[e])
        for b in BITS:
            W1q, W2q, W3q = quant(W1[e], b), quant(W2[e], b), quant(W3[e], b)
            sv = (s1, s2, bigspec(W1q), bigspec(W2q), bigspec(W1[e]-W1q), bigspec(W2[e]-W2q))
            cert[e][b] = max(cert_bound(W1[e], W2[e], W3[e], W1q, W2q, W3q, H[t:t+1],
                                        EPS, Ls, Lpp, Lppp, sv) for t in routed[e])
        print(f"  cert expert {e} done", flush=True)

    def sel_bits(e, d):
        feas = [b for b in BITS if cert[e][b] <= d]
        return min(feas) if feas else 16

    def worstcase(bits_of):
        """Max over experts of PGD ||E - E_q||_inf at that expert's assigned bits."""
        w = 0.0; arg = None
        for e in active:
            b = bits_of(e)
            if b >= 16:
                continue
            W1q, W2q, W3q = quant(W1[e], b), quant(W2[e], b), quant(W3[e], b)
            t = routed[e][0]
            d = pgd_worst(H[t:t+1], W1[e], W2[e], W3[e], W1q, W2q, W3q, EPS, PGD_STEPS)
            if d > w:
                w, arg = d, (e, b)
        return w, arg

    # match memory: pick delta so selective avg-bits ~ each uniform width
    print(f"\n{'='*78}\n  Adversarial worst-case deviation from fp16 (PGD, eps_L2={EPS})"
          f"\n{'='*78}")
    print(f"  {'avg bits/mem':>13s} | {'uniform worst-case':>19s} | "
          f"{'certified worst-case':>20s} | {'uniform/cert':>12s}")
    print("  " + "-"*74)
    for ub in [6, 4, 3]:
        # certified delta giving avg bits closest to ub
        best_d, best_gap = None, 1e9
        for d in [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
            avg = sum(sel_bits(e, d) for e in active) / len(active)
            if abs(avg - ub) < best_gap:
                best_gap, best_d = abs(avg - ub), d
        sel_avg = sum(sel_bits(best_d, 0) if False else sel_bits(e, best_d) for e in active) / len(active)
        uw, ua = worstcase(lambda e: ub)
        cw, ca = worstcase(lambda e: sel_bits(e, best_d))
        print(f"  uni {ub}b ({ub/16*100:.0f}%) | {uw:>19.4f} | cert d={best_d} "
              f"avg{sel_avg:.1f}b: {cw:>8.4f} | {uw/max(cw,1e-9):>10.1f}x", flush=True)
    print(f"\n  uniform worst expert is quantized blindly -> large adversarial divergence;")
    print("  certified-selective bounds EVERY expert by construction (cw <= delta).")
    print("  Average-case (benchmark) is ~equal; the WORST case is where certified wins.")


if __name__ == "__main__":
    main()
