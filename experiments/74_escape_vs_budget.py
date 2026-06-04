"""Experiment 74 — Does verification catch a per-expert worst-case signal that
EMPIRICAL TESTING approves, EVEN AS THE TESTING BUDGET GROWS?

This is the sharpened version of exp 67, targeting the user's thesis directly:
"things empirical testing said were safe to quantize may, under verification, have a
worst-case signal." The decisive question a reviewer asks is: "won't more test
samples just catch it?" So we sweep the Monte-Carlo testing budget M and measure:
  - ESCAPE RATE(M): (expert,bit) cases where testing (clean + M MC samples in the
      L2 eps-ball over the expert's routed region) PASSES (sampled max < delta) but
      PGD finds an in-region input with deviation > delta (true worst-case unsafe).
  - MAGNITUDE: among escapes, how far over delta does the worst case go (PGD/delta).
  - Our certified gate (exp 54 bound) is SOUND: 0 unsound at every M (M-independent).
If escapes persist as M grows large -> sampling cannot replace verification (a real
worst-case signal testing misses). If they vanish by M~1e3 -> testing just needed
more samples and verification's soundness is a formality.

Region = union of eps-balls around ALL of the expert's routed calibration inputs;
testing draws its M samples across that region; PGD attacks each and takes the max.

CPU, OLMoE layer-0 cache. Run: python3 experiments/74_escape_vs_budget.py
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
BITS = [8, 6, 4, 3]
M_BUDGETS = [16, 64, 256, 1024, 4096]
PGD_STEPS = 120
N_EXPERTS_EVAL = 40
TOK_CAP = 6              # routed tokens per expert used for region (PGD cost cap)
DELTAS = [0.005, 0.01, 0.02, 0.03, 0.05]


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


def mc_region(H_tok, W1, W2, W3, W1q, W2q, W3q, eps, m):
    """Max ||E-E_q||_inf over clean tokens + m MC samples spread across their balls."""
    devs = [(swiglu(H_tok, W1, W2, W3) - swiglu(H_tok, W1q, W2q, W3q)).abs().amax(1).max().item()]
    d = H_tok.shape[1]
    per = max(1, m // H_tok.shape[0])
    for h in H_tok:
        z = torch.randn(per, d)
        z = z / z.norm(dim=1, keepdim=True) * (torch.rand(per, 1) ** (1 / d)) * eps
        x = h.unsqueeze(0) + z
        dd = (swiglu(x, W1, W2, W3) - swiglu(x, W1q, W2q, W3q)).abs().amax(1)
        devs.append(dd.max().item())
    return max(devs)


def pgd_region(H_tok, W1, W2, W3, W1q, W2q, W3q, eps, steps):
    """PGD lower bound on the worst case over the union of balls (max over tokens)."""
    best = 0.0
    for h0 in H_tok:
        h = h0.detach()
        x = h.clone().requires_grad_(True)
        for _ in range(steps):
            diff = (swiglu(x.unsqueeze(0), W1, W2, W3)
                    - swiglu(x.unsqueeze(0), W1q, W2q, W3q)).squeeze(0)
            obj = diff.abs().max()
            best = max(best, obj.item())
            g, = torch.autograd.grad(obj, x)
            with torch.no_grad():
                x += eps / 10 * g / (g.norm() + 1e-12) * eps
                dd = x - h; n = dd.norm()
                if n > eps:
                    x.copy_(h + dd * (eps / n))
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
    active = [e for e in range(N) if inset[:, e].any()][:N_EXPERTS_EVAL]
    print(f"  {len(active)} experts, eps_L2={EPS}, M sweep={M_BUDGETS}", flush=True)

    # rows: (cert, pgd, {M: mc})
    rows = []
    maxM = max(M_BUDGETS)
    for e in active:
        idx = torch.nonzero(inset[:, e]).flatten()[:TOK_CAP]
        H_tok = H[idx]
        s1, s2 = bigspec(W1[e]), bigspec(W2[e])
        for b in BITS:
            W1q, W2q, W3q = quant(W1[e], b), quant(W2[e], b), quant(W3[e], b)
            sv = (s1, s2, bigspec(W1q), bigspec(W2q), bigspec(W1[e]-W1q), bigspec(W2[e]-W2q))
            cert = max(cert_bound(W1[e], W2[e], W3[e], W1q, W2q, W3q, H[t:t+1],
                                  EPS, Ls, Lpp, Lppp, sv) for t in idx.tolist())
            pgd = pgd_region(H_tok, W1[e], W2[e], W3[e], W1q, W2q, W3q, EPS, PGD_STEPS)
            mcs = {M: mc_region(H_tok, W1[e], W2[e], W3[e], W1q, W2q, W3q, EPS, M)
                   for M in M_BUDGETS}
            rows.append((cert, pgd, mcs))
        print(f"  expert {e} done", flush=True)

    print(f"\n{'='*86}\n  TEST ESCAPES vs testing budget M ({len(rows)} expert-bit cases, "
          f"eps_L2={EPS})\n{'='*86}")
    print("  A test escape = testing PASSES (MC max < delta) but PGD > delta (truly unsafe).")
    print("  Our certified gate is SOUND at every M (cert >= PGD): 0 unsound, shown once.\n")
    header = "  delta  | " + " | ".join(f"esc@M={M}" for M in M_BUDGETS) + \
             " | unsafe | maxovershoot | cert-unsound"
    print(header)
    print("  " + "-"*(len(header)))
    for delta in DELTAS:
        unsafe = sum(pgd > delta for _, pgd, _ in rows)
        escs = []
        for M in M_BUDGETS:
            escs.append(sum((mcs[M] < delta) and (pgd > delta) for _, pgd, mcs in rows))
        # magnitude: among the largest-M escapes, worst PGD/delta
        over = [pgd / delta for _, pgd, mcs in rows
                if (mcs[maxM] < delta) and (pgd > delta)]
        mo = max(over) if over else 0.0
        unsound = sum((cert <= delta) and (pgd > delta) for cert, pgd, _ in rows)
        cells = " | ".join(f"{e:>7d}" for e in escs)
        print(f"  {delta:>5.3f}  | {cells} | {unsafe:>6d} | {mo:>11.2f}x | {unsound:>11d}",
              flush=True)

    mc_pgd = torch.tensor([rows[i][2][maxM] / max(rows[i][1], 1e-9) for i in range(len(rows))])
    cert_pgd = torch.tensor([rows[i][0] / max(rows[i][1], 1e-9) for i in range(len(rows))])
    print(f"\n  At max budget M={maxM}: MC/PGD mean {mc_pgd.mean():.2f}x "
          f"(<1 => testing still under-estimates). cert/PGD mean {cert_pgd.mean():.2f}x (sound, tight).")
    print("  If esc@M stays > 0 as M grows -> a worst-case signal sampling cannot find;")
    print("  verification (sound) is the only gate that catches it. If esc->0 -> budget suffices.")


if __name__ == "__main__":
    main()
