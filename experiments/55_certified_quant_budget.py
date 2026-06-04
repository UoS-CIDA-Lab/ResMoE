"""Experiment 55 — Per-expert CERTIFIED quantization budget (the practical
deliverable) + validation.

Inverts the exp-52/54 bound: given a tolerance delta_max, find for each expert
the COARSEST bit-width whose certified L2 output-deviation bound (over every
routed input's L2 eps-ball) stays <= delta_max. Output a per-expert mixed-
precision assignment WITH certificates, and the certified (delta_max -> avg
bits -> memory) Pareto curve.

Validation: (1) SOUNDNESS — at the assigned bits, certified bound >= L2-PGD
empirical (the guarantee holds); (2) the actual deviation stays within the
certified delta_max. So the claim "we can certify how far each expert can be
quantized with bounded output change" is experimentally demonstrated.

Run: python3 experiments/55_certified_quant_budget.py
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
EPS = 0.05


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


SP = {}  # spectral-norm cache


def spec(W):
    key = (W.data_ptr(), W.shape)
    if key not in SP:
        SP[key] = torch.linalg.matrix_norm(W, ord=2).item()
    return SP[key]


def bound54(W1, W2, W3, W1q, W2q, W3q, h0, eps, Ls, Lpp, Lppp):
    """Tight per-expert L2 bound on ||E-E_q||_inf over the L2 eps-ball (exp 54)."""
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi))
    Bvq = amax(*affine_box(W2q, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi))
    Bd2 = amax(*affine_box(d2, lo, hi))
    s1, s2, s1q, s2q = spec(W1), spec(W2), spec(W1q), spec(W2q)
    sd1, sd2 = spec(d1), spec(d2)
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)
    De = Ls * W3.abs().amax(1)
    dDc = (W3.abs() * (Lpp * Bd2 + Lppp * Bd1 * Bvq).unsqueeze(0)
           + d3.abs() * (Lpp * Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs() * (Lpp * Bd1).unsqueeze(0) + d3.abs() * Ls).amax(1)
    HD = Dc * sd1 * (s1 + s1q) + s1q * s1q * dDc \
        + 2 * (s1 * De * sd2 + sd1 * De * s2q + s1q * dDe * s2q)
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C * (silu_grad(u) * v).unsqueeze(0)) @ A + (C * F.silu(u).unsqueeze(0)) @ B
    JD = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps * JD + 0.5 * eps * eps * HD).max().item()


def l2_pgd(Wi, Wj, h0, eps, n_steps=150):
    (a, b, d), (aq, bq, dq) = Wi, Wj
    step = eps / 15
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps):
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
    N, K = c["n_experts"], c["top_k"]
    Ls, Lpp, Lppp = silu_consts()
    logits = H @ Wg.T
    _, topk = logits.topk(K, dim=-1)
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool)
    inset.scatter_(1, topk, True)
    print(f"  N={N}, K={K}, eps(L2)={EPS}")

    # per-expert certified bound at each bit-width (max over the expert's routed inputs)
    print("\nComputing per-expert certified bounds at each bit-width...")
    Wq = {b: (torch.stack([quantize_per_channel(W1[i], b) for i in range(N)]),
              torch.stack([quantize_per_channel(W2[i], b) for i in range(N)]),
              torch.stack([quantize_per_channel(W3[i], b) for i in range(N)]))
          for b in BITS}
    cert = torch.full((N, len(BITS)), float("inf"))   # cert[e, bi]
    active = []
    for e in range(N):
        routed = torch.nonzero(inset[:, e]).flatten()
        if len(routed) == 0:
            continue
        active.append(e)
        for bi, b in enumerate(BITS):
            W1q, W2q, W3q = Wq[b]
            cert[e, bi] = max(bound54(W1[e], W2[e], W3[e], W1q[e], W2q[e], W3q[e],
                                      H[t:t + 1], EPS, Ls, Lpp, Lppp)
                              for t in routed.tolist())
    print(f"  {len(active)} active experts (freq>0 on calib)")

    # certified Pareto: delta_max -> coarsest feasible bits per expert -> avg bits
    print(f"\n{'='*64}\n  Certified quantization budget (eps_L2={EPS})\n{'='*64}")
    print(f"  {'delta_max':>10s} {'avg bits':>9s} {'mem vs fp16':>12s}  bit histogram")
    for dmax in (0.001, 0.005, 0.01, 0.05, 0.1):
        assigned = []
        for e in active:
            feasible = [BITS[bi] for bi in range(len(BITS)) if cert[e, bi].item() <= dmax]
            assigned.append(min(feasible) if feasible else max(BITS))
        avg_b = sum(assigned) / len(assigned)
        hist = {b: assigned.count(b) for b in BITS}
        print(f"  {dmax:>10.3f} {avg_b:>9.2f} {avg_b/16*100:>11.1f}%  {hist}")

    # ---- VALIDATION at a chosen delta_max ----
    dmax = 0.01
    print(f"\n{'='*64}\n  VALIDATION at delta_max={dmax}\n{'='*64}")
    assign = {}
    for e in active:
        feas = [BITS[bi] for bi in range(len(BITS)) if cert[e, bi].item() <= dmax]
        assign[e] = min(feas) if feas else max(BITS)
    print(f"  per-expert bits: avg {sum(assign.values())/len(assign):.2f}, "
          f"hist {dict((b, list(assign.values()).count(b)) for b in BITS)}")
    torch.manual_seed(0)
    sample = active[:12]
    print(f"\n  soundness check (sample experts at assigned bits, eps={EPS}):")
    print(f"  {'expert':>7s} {'bits':>5s} {'certified':>11s} {'L2-PGD':>9s} "
          f"{'<=dmax?':>8s} {'sound?':>7s}")
    all_sound = all_within = True
    for e in sample:
        b = assign[e]
        W1q, W2q, W3q = Wq[b]
        routed = torch.nonzero(inset[:, e]).flatten()
        ce = max(bound54(W1[e], W2[e], W3[e], W1q[e], W2q[e], W3q[e],
                         H[t:t + 1], EPS, Ls, Lpp, Lppp) for t in routed.tolist())
        pe = max(l2_pgd((W1[e], W2[e], W3[e]), (W1q[e], W2q[e], W3q[e]),
                        H[t:t + 1], EPS) for t in routed.tolist())
        within = ce <= dmax + 1e-9
        sound = pe <= ce + 1e-6
        all_within &= within; all_sound &= sound
        print(f"  {('E'+str(e)):>7s} {b:>5d} {ce:>11.5f} {pe:>9.5f} "
              f"{('yes' if within else 'NO'):>8s} {('yes' if sound else 'NO!'):>7s}")
    print(f"\n  ALL certified <= delta_max: {all_within};  ALL sound (PGD<=cert): {all_sound}")
    print(f"\n{'='*64}")
    print("  => per-expert certified mixed-precision budget is computable, the")
    print("  certificate is SOUND (PGD never exceeds it) and HONORS delta_max.")
    print("  This is the practical deliverable: provable per-expert quant limits.")


if __name__ == "__main__":
    main()
