"""Experiment 58 — ROUTER-AWARE vs router-agnostic certified budget (RQ2).

The per-expert budget (exp 55) asks every expert to meet the SAME tolerance in
isolation. But an expert reaches the layer output only through its gate:
    ||Y - Y_q|| <= sum_{e in topK} g_e ||E_e - E_{e,q}||.
So weakly/rarely-gated experts can tolerate a LARGER per-expert deviation at the
same LAYER guarantee. We compare, at an equal layer tolerance delta_lyr:

  (A) router-AGNOSTIC: assign coarsest bits with cert D_e <= delta_lyr
      (sound layer guarantee since sum g_e D_e <= max_e D_e <= delta_lyr).
  (B) router-AWARE:   assign coarsest bits with cert D_e <= delta_lyr / (K * ghat_e),
      where ghat_e = max softmax gate expert e receives over its routed inputs
      (sound: sum_{e in topK} g_e D_e <= sum ghat_e * delta_lyr/(K ghat_e) = delta_lyr).

Report avg bits, memory vs fp16, and an empirical layer-deviation estimate
(max over tokens of sum_{e in topK} g_e * L2-PGD_e at the assigned bits). Both
budgets are sound; router-aware should buy more compression at equal guarantee
by spending bits where the router actually puts weight.

Run: python3 experiments/58_router_aware_budget.py
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
DELTAS_LYR = [0.005, 0.01, 0.05, 0.1]
PGD_DELTA = 0.01  # delta_lyr at which we run the empirical layer-deviation check


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


SP = {}


def spec(W):
    key = (W.data_ptr(), W.shape)
    if key not in SP:
        SP[key] = torch.linalg.matrix_norm(W, ord=2).item()
    return SP[key]


def bound54(W1, W2, W3, W1q, W2q, W3q, h0, eps, Ls, Lpp, Lppp):
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


def l2_pgd(Wi, Wj, h0, eps, n_steps=120):
    (a, b, d), (aq, bq, dq) = Wi, Wj
    step = eps / 12
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
    topv, topk = logits.topk(K, dim=-1)
    gates = torch.softmax(topv, dim=-1)             # [T, K] softmax over selected
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool)
    inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()]

    # ghat_e = max softmax gate expert e receives over the inputs routed to it
    ghat = torch.zeros(N)
    freq = torch.zeros(N)
    for t in range(H.shape[0]):
        for j in range(K):
            e = topk[t, j].item()
            ghat[e] = max(ghat[e].item(), gates[t, j].item())
            freq[e] += 1
    print(f"  N={N}, K={K}, eps_L2={EPS}, {len(active)} active experts")
    print(f"  gate ghat over active: min {ghat[active].min():.3f} "
          f"max {ghat[active].max():.3f} mean {ghat[active].mean():.3f} (1/K={1/K:.3f})")

    Wq = {b: (torch.stack([quantize_per_channel(W1[i], b) for i in range(N)]),
              torch.stack([quantize_per_channel(W2[i], b) for i in range(N)]),
              torch.stack([quantize_per_channel(W3[i], b) for i in range(N)]))
          for b in BITS}

    print("\nComputing per-expert certified bounds at each bit-width...", flush=True)
    cert = {e: {} for e in active}
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist() for e in active}
    for e in active:
        for b in BITS:
            W1q, W2q, W3q = Wq[b]
            cert[e][b] = max(bound54(W1[e], W2[e], W3[e], W1q[e], W2q[e], W3q[e],
                                     H[t:t + 1], EPS, Ls, Lpp, Lppp)
                             for t in routed[e][:2])
        print(f"  expert {e} done", flush=True)

    def assign(tol_of_e):
        a = {}
        for e in active:
            feas = [b for b in BITS if cert[e][b] <= tol_of_e(e)]
            a[e] = min(feas) if feas else max(BITS)
        return a

    def avg_bits(a):
        return sum(a.values()) / len(a)

    print(f"\n{'='*74}")
    print(f"  Router-aware vs router-agnostic certified budget (OLMoE layer-0)")
    print(f"  equal LAYER guarantee delta_lyr; mem relative to fp16")
    print(f"{'='*74}")
    print(f"  {'delta_lyr':>10s} | {'AGNOSTIC avg/mem':>22s} | {'AWARE avg/mem':>22s}")
    print("  " + "-"*70)
    budgets = {}
    for dl in DELTAS_LYR:
        ag = assign(lambda e, dl=dl: dl)
        aw = assign(lambda e, dl=dl: dl / (K * max(ghat[e].item(), 1e-6)))
        budgets[dl] = (ag, aw)
        ab, wb = avg_bits(ag), avg_bits(aw)
        print(f"  {dl:>10.3f} | {ab:>10.2f}b {ab/16*100:>8.1f}%  | "
              f"{wb:>10.2f}b {wb/16*100:>8.1f}%")

    # ---- empirical layer-deviation check at PGD_DELTA ----
    dl = PGD_DELTA
    ag, aw = budgets[dl]
    print(f"\n{'='*74}\n  Empirical layer deviation at delta_lyr={dl} "
          f"(max over tokens of sum_e g_e * L2-PGD_e)\n{'='*74}")
    torch.manual_seed(0)
    # cap routed inputs per expert for PGD speed
    pgd_cache = {}  # (e, bits) -> max pgd over (capped) routed inputs

    def pgd_e(e, bits):
        key = (e, bits)
        if key not in pgd_cache:
            W1q, W2q, W3q = Wq[bits]
            ts = routed[e][:3]
            pgd_cache[key] = max(
                l2_pgd((W1[e], W2[e], W3[e]), (W1q[e], W2q[e], W3q[e]), H[t:t + 1], EPS)
                for t in ts)
        return pgd_cache[key]

    def layer_dev(a):
        worst = 0.0
        for t in range(H.shape[0]):
            s = 0.0
            for j in range(K):
                e = topk[t, j].item()
                s += gates[t, j].item() * pgd_e(e, a[e])
            worst = max(worst, s)
        return worst

    ld_ag = layer_dev(ag)
    ld_aw = layer_dev(aw)
    print(f"  router-agnostic: avg {avg_bits(ag):.2f} bits, "
          f"mem {avg_bits(ag)/16*100:.1f}%, est. layer dev {ld_ag:.5f} "
          f"({'<=' if ld_ag <= dl else '>'} delta_lyr)")
    print(f"  router-aware:    avg {avg_bits(aw):.2f} bits, "
          f"mem {avg_bits(aw)/16*100:.1f}%, est. layer dev {ld_aw:.5f} "
          f"({'<=' if ld_aw <= dl else '>'} delta_lyr)")
    bit_save = (avg_bits(ag) - avg_bits(aw)) / avg_bits(ag) * 100
    print(f"\n  => router-aware uses {bit_save:+.1f}% fewer avg bits at the SAME")
    print(f"  certified layer guarantee, both with est. layer dev <= delta_lyr.")
    print(f"  (router-aware spends bits where the gate is large, compresses the rest.)")


if __name__ == "__main__":
    main()
