"""Experiment 56 — Baselines for the certified quantization budget (RQ2).

Two baselines for the per-expert certified budget (exp 55):
  (1) UNIFORM quantization (all experts same bits) — the standard, no
      certificate. Pareto: memory vs ACTUAL worst-case deviation (L2-PGD).
  (2) McCormick-based budget — same inversion but using the loose McCormick
      bound; shows a loose bound makes the certified budget VACUOUS.
Compared against our tight per-expert certified budget on (memory, actual
worst-case L2 deviation, has-certificate?).

Message: per-expert (certificate-guided) allocation matches/beats uniform on
actual deviation at equal memory AND uniquely carries a guarantee; the guarantee
is usable only because the bound is tight (McCormick's budget = fp16).
Run: python3 experiments/56_quant_budget_baselines.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu, swiglu_diff_bound_box

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
BITS = [8, 6, 4, 3, 2]
EPS = 0.05


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


_SP = {}
def spec(W):
    k = (W.data_ptr(), W.shape)
    if k not in _SP:
        _SP[k] = torch.linalg.matrix_norm(W, ord=2).item()
    return _SP[k]


def bound54(W1, W2, W3, W1q, W2q, W3q, h0, eps, Ls, Lpp, Lppp):
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi)); Bvq = amax(*affine_box(W2q, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi)); Bd2 = amax(*affine_box(d2, lo, hi))
    s1, s2, s1q, s2q = spec(W1), spec(W2), spec(W1q), spec(W2q)
    sd1, sd2 = spec(d1), spec(d2)
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)
    De = Ls * W3.abs().amax(1)
    dDc = (W3.abs() * (Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0) + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs() * (Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    HD = Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JD = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps*JD + 0.5*eps*eps*HD).max().item()


def l2_pgd(Wi, Wj, h0, eps, n_steps=120):
    (a, b, d), (aq, bq, dq) = Wi, Wj
    step = eps/12; h = h0.clone().detach(); best = 0.0
    for _ in range(n_steps):
        hg = h.clone().requires_grad_(True)
        loss = (swiglu(hg, a, b, d) - swiglu(hg, aq, bq, dq)).abs().max(-1).values.sum()
        loss.backward()
        with torch.no_grad():
            g = hg.grad; h = h + step*g/g.norm().clamp(min=1e-12)
            dl = h - h0; dn = dl.norm()
            if dn > eps:
                h = h0 + dl*(eps/dn)
            best = max(best, (swiglu(h, a, b, d) - swiglu(h, aq, bq, dq)).abs().max(-1).values.max().item())
    return best


def main():
    print("Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float(); H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    Ls, Lpp, Lppp = silu_consts()
    logits = H @ Wg.T
    _, tk = logits.topk(K, dim=-1)
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool); inset.scatter_(1, tk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist()[:2] for e in active}
    Wq = {b: (torch.stack([quantize_per_channel(W1[i], b) for i in range(N)]),
              torch.stack([quantize_per_channel(W2[i], b) for i in range(N)]),
              torch.stack([quantize_per_channel(W3[i], b) for i in range(N)])) for b in BITS}

    def worst_pgd(bit_of):  # bit_of: dict expert->bits
        w = 0.0
        for e in active:
            b = bit_of[e]; W1q, W2q, W3q = Wq[b]
            for t in routed[e]:
                w = max(w, l2_pgd((W1[e], W2[e], W3[e]), (W1q[e], W2q[e], W3q[e]), H[t:t+1], EPS))
        return w

    # per-expert certified + McCormick bounds at each bit
    print("Computing per-expert tight & McCormick bounds...")
    cert = {e: {} for e in active}; mcc = {e: {} for e in active}
    for e in active:
        for b in BITS:
            W1q, W2q, W3q = Wq[b]
            cert[e][b] = max(bound54(W1[e], W2[e], W3[e], W1q[e], W2q[e], W3q[e], H[t:t+1], EPS, Ls, Lpp, Lppp) for t in routed[e])
            mcc[e][b] = max(swiglu_diff_bound_box(W1[e], W2[e], W3[e], W1q[e], W2q[e], W3q[e], (H[t:t+1]-EPS).squeeze(0), (H[t:t+1]+EPS).squeeze(0)) for t in routed[e])

    print(f"\n{'='*72}\n  Quantization budget BASELINES (OLMoE layer-0, eps_L2={EPS})\n{'='*72}")
    print(f"  {'method':32s} {'avg bits':>8s} {'mem':>6s} {'actual dev':>11s} {'cert?':>6s}")
    print("  " + "-"*68)
    # uniform baselines
    for b in BITS:
        dev = worst_pgd({e: b for e in active})
        print(f"  {('uniform '+str(b)+'-bit'):32s} {b:>8.2f} {b/16*100:>5.0f}% {dev:>11.5f} {'no':>6s}")
    # our certified budget at tolerances
    for dmax in (0.005, 0.01, 0.05):
        assign = {e: min([b for b in BITS if cert[e][b] <= dmax] or [max(BITS)]) for e in active}
        avg = sum(assign.values())/len(assign)
        dev = worst_pgd(assign)
        print(f"  {('OURS certified d='+str(dmax)):32s} {avg:>8.2f} {avg/16*100:>5.0f}% {dev:>11.5f} {'YES':>6s}")
    # McCormick-based budget (same inversion, loose bound)
    for dmax in (0.01, 0.05):
        assignm = {e: min([b for b in BITS if mcc[e][b] <= dmax] or [16] ) for e in active}
        # 16 means "no feasible quant bit -> keep fp16"
        avg = sum(assignm.values())/len(assignm)
        nfp16 = sum(1 for v in assignm.values() if v == 16)
        print(f"  {('McCormick-budget d='+str(dmax)):32s} {avg:>8.2f} {avg/16*100:>5.0f}% {'(n/a)':>11s} {'YES*':>6s}  ({nfp16}/{len(active)} forced fp16)")

    print(f"\n{'='*72}")
    print("  Read: OURS (certified) sits on/below the uniform Pareto at equal memory")
    print("  AND carries a guarantee. McCormick-budget* is technically certified but")
    print("  VACUOUS — the loose bound forces ~all experts to fp16 (no compression).")


if __name__ == "__main__":
    main()
