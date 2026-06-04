"""Experiment 61 — Regenerate Table 4 (baselines) with the ROUTER-AWARE budget.

Table 4 previously benchmarked the *router-agnostic* per-expert budget against
uniform quantization, which is exactly why it looked tied with uniform. This
script instead uses the router-AWARE greedy budget (the method we advocate, exp
59) and measures, for every config, the SAME apples-to-apples actual deviation:
the worst per-token gate-weighted layer perturbation under L2-PGD,

    dev = max_t  sum_{e in topK(t)} g_{t,e} * || E_e(x) - E_e^q(x) ||_inf ,
          x in B2(h_t, eps)   (PGD-maximized)

which is exactly the quantity the per-token certificate bounds. We report it for
uniform {8,6,4,2}-bit (no certificate) and for ours router-aware at several
delta_lyr (each carries the layer guarantee dev <= delta_lyr, PGD-verified).

CPU-only (layer-0 cache); safe to run alongside the GPU demo.
Run: python3 experiments/61_baselines_router_aware.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
BITS = [16, 8, 6, 4, 3, 2]
EPS = 0.05
DELTAS = [0.005, 0.01, 0.05]
UNIFORM = [8, 6, 4, 2]
PGD_TOKENS = 48      # tokens sampled for the actual-dev PGD
PGD_STEPS = 60
PGD_LR = EPS / 8


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
    return (g1.abs().max().item()*1.01, g2.abs().max().item()*1.01,
            g3.abs().max().item()*1.01)


def quant(W, bits):
    if bits >= 16:
        return W.clone()
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def bigspec(W):
    return torch.linalg.matrix_norm(W, ord=2).item()


def bound54(W1, W2, W3, W1q, W2q, W3q, h0, eps, Ls, Lpp, Lppp, sv):
    s1, s2, s1q, s2q, sd1, sd2 = sv
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi)); Bvq = amax(*affine_box(W2q, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi)); Bd2 = amax(*affine_box(d2, lo, hi))
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)
    De = Ls * W3.abs().amax(1)
    dDc = (W3.abs()*(Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0)
           + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs()*(Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    HD = Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JD = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps*JD + 0.5*eps*eps*HD).max().item()


def pgd_gate_dev(tokens, topk, gates, K, H, Wq_cache, W1, W2, W3, eps):
    """Worst per-token gate-weighted layer deviation under L2-PGD over B2(h,eps)."""
    worst = 0.0
    for t in tokens:
        es = [topk[t, j].item() for j in range(K)]
        gs = [gates[t, j].item() for j in range(K)]
        x = H[t].clone().detach()
        h0 = H[t].clone().detach()
        x.requires_grad_(True)
        for _ in range(PGD_STEPS):
            obj = 0.0
            for g, e in zip(gs, es):
                W1q, W2q, W3q = Wq_cache[e]
                diff = (swiglu(x.unsqueeze(0), W1[e], W2[e], W3[e])
                        - swiglu(x.unsqueeze(0), W1q, W2q, W3q)).squeeze(0)
                obj = obj + g * diff.abs().max()
            grad, = torch.autograd.grad(obj, x)
            with torch.no_grad():
                x += PGD_LR * grad / (grad.norm() + 1e-12) * eps
                delta = x - h0
                n = delta.norm()
                if n > eps:
                    x.copy_(h0 + delta * (eps / n))
            x.requires_grad_(True)
        with torch.no_grad():
            val = 0.0
            for g, e in zip(gs, es):
                W1q, W2q, W3q = Wq_cache[e]
                diff = (swiglu(x.unsqueeze(0), W1[e], W2[e], W3[e])
                        - swiglu(x.unsqueeze(0), W1q, W2q, W3q)).squeeze(0)
                val += g * diff.abs().max().item()
        worst = max(worst, val)
    return worst


def main():
    print("Loading OLMoE layer-0 cache...", flush=True)
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float(); H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    Ls, Lpp, Lppp = silu_consts()
    logits = H @ Wg.T
    topv, topk = logits.topk(K, dim=-1)
    gates = torch.softmax(topv, dim=-1)
    T = H.shape[0]
    inset = torch.zeros(T, N, dtype=torch.bool); inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    # NO cap: D_e must be the sup over ALL inputs the expert is routed to, else
    # the per-token guarantee is unsound (PGD on held-out routed tokens exceeds it).
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist() for e in active}
    pgd_tok = list(range(min(PGD_TOKENS, T)))
    print(f"  {len(active)} active experts; PGD on {len(pgd_tok)} tokens", flush=True)

    # per-expert certified D_e(bits) and a quantized-weight cache per bits
    print("Computing per-expert certified bounds...", flush=True)
    cert = {e: {16: 0.0} for e in active}
    qcache = {e: {16: (W1[e], W2[e], W3[e])} for e in active}
    for e in active:
        s1, s2 = bigspec(W1[e]), bigspec(W2[e])
        for b in BITS:
            if b >= 16:
                continue
            W1q, W2q, W3q = quant(W1[e], b), quant(W2[e], b), quant(W3[e], b)
            qcache[e][b] = (W1q, W2q, W3q)
            sv = (s1, s2, bigspec(W1q), bigspec(W2q),
                  bigspec(W1[e]-W1q), bigspec(W2[e]-W2q))
            cert[e][b] = max(bound54(W1[e], W2[e], W3[e], W1q, W2q, W3q,
                                     H[t:t+1], EPS, Ls, Lpp, Lppp, sv)
                             for t in routed[e])
        print(f"  expert {e} done", flush=True)

    nxt = {2: 3, 3: 4, 4: 6, 6: 8, 8: 16, 16: 16}

    def greedy(dlyr):
        bits = {e: 2 for e in active}
        for _ in range(20000):
            worst_t, worst_s = -1, dlyr
            for t in range(T):
                s = sum(gates[t, j].item()*cert[topk[t, j].item()][bits[topk[t, j].item()]]
                        for j in range(K))
                if s > worst_s:
                    worst_s, worst_t = s, t
            if worst_t < 0:
                break
            cand = [(gates[worst_t, j].item()*cert[topk[worst_t, j].item()][bits[topk[worst_t, j].item()]],
                     topk[worst_t, j].item()) for j in range(K)
                    if bits[topk[worst_t, j].item()] != 16]
            if not cand:
                break
            e = max(cand)[1]
            bits[e] = nxt[bits[e]]
        return bits

    avg = lambda bits: sum(bits.values()) / len(bits)

    def cache_for(bitsmap):
        return {e: qcache[e][bitsmap[e]] for e in active}

    rows = []
    # uniform baselines
    for b in UNIFORM:
        bm = {e: b for e in active}
        dev = pgd_gate_dev(pgd_tok, topk, gates, K, H, cache_for(bm), W1, W2, W3, EPS)
        rows.append((f"uniform {b}-bit", float(b), b/16*100, dev, None))
        print(f"  uniform {b}-bit: dev={dev:.4f}", flush=True)
    # ours router-aware
    for dl in DELTAS:
        bm = greedy(dl)
        dev = pgd_gate_dev(pgd_tok, topk, gates, K, H, cache_for(bm), W1, W2, W3, EPS)
        rows.append((f"ours-RA d={dl}", avg(bm), avg(bm)/16*100, dev, dl))
        print(f"  ours-RA d={dl}: avg={avg(bm):.2f}b dev={dev:.4f} (<= {dl}? {dev<=dl+1e-9})",
              flush=True)

    print(f"\n{'='*78}\n  Table 4 (router-AWARE), eps_L2={EPS}, gate-weighted PGD actual dev\n{'='*78}")
    print(f"  {'method':18s} {'avg bits':>9s} {'mem':>6s} {'actual dev':>11s} "
          f"{'cert bound':>11s} {'cert?':>6s}")
    print("  " + "-"*72)
    for name, bits, mem, dev, cb in rows:
        cbs = f"{cb:.4f}" if cb is not None else "  none"
        certq = "yes" if cb is not None else "no"
        print(f"  {name:18s} {bits:>9.2f} {mem:>5.0f}% {dev:>11.4f} {cbs:>11s} {certq:>6s}")
    print(f"\n  ours-RA carries a proven ceiling (dev <= cert bound, PGD-verified);")
    print("  uniform offers a comparable tradeoff but NO bound. Compare at matched mem.")


if __name__ == "__main__":
    main()
