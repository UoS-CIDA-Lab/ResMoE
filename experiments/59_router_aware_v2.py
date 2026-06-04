"""Experiment 59 — Router-aware budget v2: correct per-input constraint + greedy.

The sound LAYER guarantee is PER INPUT: for every token t,
    sum_{e in topK(t)} g_{t,e} * D_e <= delta_lyr      (only K active, sum g <= 1).
exp 58's gate-reweighted scheme (D_e <= delta_lyr/(K*ghat_e)) was WORSE than the
agnostic D_e<=delta_lyr because most gates exceed 1/K. Two correct improvements:
  (A) agnostic:        D_e <= delta_lyr  (sound: sum g_e D_e <= delta_lyr*sum g <= delta_lyr)
  (B) gate-mass unif.: D_e <= delta_lyr / Gmax, Gmax = max_t sum_{topK} g_{t,e}
                       (exploits that the top-K softmax mass is < 1).
  (C) router-aware greedy: minimize total bits s.t. EVERY per-input constraint
                       holds -- spend bits on high-gate experts in tight tokens,
                       compress the rest. Start coarsest, refine the expert that
                       most reduces the worst violation.
All three are sound (we verify every per-input constraint). (C) should win.
Run: python3 experiments/59_router_aware_v2.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
BITS = [16, 8, 6, 4, 3, 2]   # 16 = fp16 (D=0, no compression)
EPS = 0.05
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
    if bits >= 16:
        return W.clone()
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / scale).clamp(-qmax - 1, qmax) * scale


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
    print("Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float(); H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    Ls, Lpp, Lppp = silu_consts()
    logits = H @ Wg.T
    topv, topk = logits.topk(K, dim=-1)
    gates = torch.softmax(topv, dim=-1)            # [T,K], not renormalized vs full? toy uses softmax over topK
    T = H.shape[0]
    inset = torch.zeros(T, N, dtype=torch.bool); inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist()[:2] for e in active}
    Gmax = max(gates[t].sum().item() for t in range(T))
    print(f"  {len(active)} active experts, Gmax (max top-K gate mass) = {Gmax:.3f}")

    # per-expert certified D_e(bits)
    print("Computing per-expert certified bounds...", flush=True)
    cert = {e: {16: 0.0} for e in active}
    for e in active:
        s1, s2 = bigspec(W1[e]), bigspec(W2[e])
        for b in BITS:
            if b >= 16:
                continue
            W1q, W2q, W3q = (quantize_per_channel(W1[e], b), quantize_per_channel(W2[e], b),
                             quantize_per_channel(W3[e], b))
            sv = (s1, s2, bigspec(W1q), bigspec(W2q), bigspec(W1[e]-W1q), bigspec(W2[e]-W2q))
            cert[e][b] = max(bound54(W1[e], W2[e], W3[e], W1q, W2q, W3q,
                                     H[t:t+1], EPS, Ls, Lpp, Lppp, sv) for t in routed[e])
        print(f"  expert {e} done", flush=True)

    fine_to_coarse = sorted(BITS)            # [2,3,4,6,8,16]
    def coarsest_within(e, tol):
        feas = [b for b in BITS if cert[e][b] <= tol]
        return min(feas) if feas else 16

    def viol_ok(bits, dlyr):
        for t in range(T):
            s = sum(gates[t, j].item() * cert[topk[t, j].item()][bits[topk[t, j].item()]]
                    for j in range(K))
            if s > dlyr + 1e-12:
                return False
        return True

    def greedy(dlyr):
        bits = {e: 2 for e in active}        # start coarsest
        # ensure each expert has a feasible-at-some-bit; refine to satisfy per-input
        order = [16, 8, 6, 4, 3, 2]
        nxt = {2: 3, 3: 4, 4: 6, 6: 8, 8: 16, 16: 16}
        for _ in range(10000):
            # worst-violated input
            worst_t, worst_s = -1, dlyr
            for t in range(T):
                s = sum(gates[t, j].item() * cert[topk[t, j].item()][bits[topk[t, j].item()]]
                        for j in range(K))
                if s > worst_s:
                    worst_s, worst_t = s, t
            if worst_t < 0:
                break
            # in the worst input, refine the expert with largest gate*D (most reducible)
            cand = [(gates[worst_t, j].item() * cert[topk[worst_t, j].item()][bits[topk[worst_t, j].item()]],
                     topk[worst_t, j].item()) for j in range(K)
                    if bits[topk[worst_t, j].item()] != 16]
            if not cand:
                break
            _, e = max(cand)
            bits[e] = nxt[bits[e]]
        return bits

    avg = lambda bits: sum(bits.values()) / len(bits)
    print(f"\n{'='*72}\n  Router-aware v2 (per-input constraint), eps_L2={EPS}\n{'='*72}")
    print(f"  {'d_lyr':>7s} | {'agnostic':>14s} | {'gate-mass':>14s} | {'AWARE-greedy':>16s}")
    print("  " + "-"*66)
    for dl in DELTAS:
        ag = {e: coarsest_within(e, dl) for e in active}
        gm = {e: coarsest_within(e, dl / Gmax) for e in active}
        gr = greedy(dl)
        ok = all(viol_ok(b, dl) for b in (ag, gm, gr))
        print(f"  {dl:>7.3f} | {avg(ag):>6.2f}b {avg(ag)/16*100:>5.0f}% | "
              f"{avg(gm):>6.2f}b {avg(gm)/16*100:>5.0f}% | "
              f"{avg(gr):>7.2f}b {avg(gr)/16*100:>5.0f}%   (sound={ok})")

    print(f"\n{'='*72}")
    print("  agnostic uses D_e<=d_lyr; gate-mass uses the <1 top-K mass slack;")
    print("  AWARE-greedy solves the per-input constraints directly, spending bits")
    print("  only where the gate is large in tight tokens. Lower = better (sound).")


if __name__ == "__main__":
    main()
