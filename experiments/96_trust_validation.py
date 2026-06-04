"""Experiment 96 — Are the outputs we CERTIFY as trustworthy ACTUALLY trustworthy?

For "certified-safe" to mean anything, an expert-quantization our certificate clears
at tolerance delta (cert <= delta) must REALLY stay within delta of the original for
every input in the operating eps-ball -- including the worst-case ones. We stress this
with STRONG multi-restart PGD: for each (expert,bit) the certificate calls safe, search
the eps-ball hard for any input whose true deviation exceeds delta.
  certified-trustworthy is REAL iff: 0 of the cert-safe cases are PGD-violated.
We contrast with TESTING-trust (clean point + 256 MC samples <= delta): testing passes
cases that PGD then violates (test escapes) -- its "trust" is not actually trustworthy.
Reports, per delta: #cert-safe, #cert-safe PGD-violated (must be 0), tightness;
#testing-trusted, #testing-trusted PGD-violated (>0 = unfounded trust).

CPU, OLMoE layer-0 cache. Run: python3 experiments/96_trust_validation.py
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
MC_SAMPLES = 256
PGD_RESTARTS = 8
PGD_STEPS = 120
N_EXPERTS = 40
DELTAS = [0.005, 0.01, 0.02, 0.03, 0.05]


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


def mc_test(h0, W1, W2, W3, W1q, W2q, W3q, eps, m):
    h = h0.squeeze(0)
    best = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).abs().max().item()
    z = torch.randn(m, h.shape[0])
    z = z / z.norm(dim=1, keepdim=True) * (torch.rand(m, 1) ** (1/h.shape[0])) * eps
    d = (swiglu(h.unsqueeze(0) + z, W1, W2, W3) - swiglu(h.unsqueeze(0) + z, W1q, W2q, W3q)).abs().amax(1)
    return max(best, d.max().item())


def strong_pgd(h0, W1, W2, W3, W1q, W2q, W3q, eps, restarts, steps):
    h = h0.squeeze(0).detach()
    best = 0.0
    for r in range(restarts):
        if r == 0:
            x = h.clone()
        else:
            z = torch.randn_like(h); x = h + z / z.norm() * eps * torch.rand(1).item()
        x = x.requires_grad_(True)
        for _ in range(steps):
            obj = (swiglu(x.unsqueeze(0), W1, W2, W3) - swiglu(x.unsqueeze(0), W1q, W2q, W3q)).abs().max()
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
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool)
    inset.scatter_(1, (H @ Wg.T).topk(K, -1).indices, True)
    active = [e for e in range(N) if inset[:, e].any()][:N_EXPERTS]
    print(f"  {len(active)} experts, eps={EPS}, strong PGD {PGD_RESTARTS}x{PGD_STEPS}", flush=True)

    rows = []   # (clean, mc, pgd, cert)
    for e in active:
        t = torch.nonzero(inset[:, e]).flatten()[0].item(); h0 = H[t:t+1]
        s1, s2 = bigspec(W1[e]), bigspec(W2[e])
        for b in BITS:
            W1q, W2q, W3q = quant(W1[e], b), quant(W2[e], b), quant(W3[e], b)
            sv = (s1, s2, bigspec(W1q), bigspec(W2q), bigspec(W1[e]-W1q), bigspec(W2[e]-W2q))
            cert = cert_bound(W1[e], W2[e], W3[e], W1q, W2q, W3q, h0, EPS, Ls, Lpp, Lppp, sv)
            mc = mc_test(h0, W1[e], W2[e], W3[e], W1q, W2q, W3q, EPS, MC_SAMPLES)
            pgd = strong_pgd(h0, W1[e], W2[e], W3[e], W1q, W2q, W3q, EPS, PGD_RESTARTS, PGD_STEPS)
            clean = (swiglu(h0, W1[e], W2[e], W3[e]) - swiglu(h0, W1q, W2q, W3q)).abs().max().item()
            rows.append((clean, mc, pgd, cert))
        print(f"  expert {e} done", flush=True)

    print(f"\n{'='*82}\n  Is 'certified trustworthy' actually trustworthy? (strong PGD = ground"
          f" truth)\n{'='*82}")
    print(f"  {'delta':>6s} | {'#cert-safe':>10s} | {'cert VIOLATED':>13s} | "
          f"{'#test-trusted':>13s} | {'test VIOLATED':>13s}")
    print("  " + "-"*70)
    for d in DELTAS:
        csafe = [r for r in rows if r[3] <= d]              # certificate calls safe
        cviol = sum(r[2] > d for r in csafe)                # but PGD exceeds delta
        ttrust = [r for r in rows if r[1] <= d]             # testing calls safe
        tviol = sum(r[2] > d for r in ttrust)               # but PGD exceeds delta
        print(f"  {d:>6.3f} | {len(csafe):>10d} | {cviol:>13d} | {len(ttrust):>13d} | "
              f"{tviol:>13d}", flush=True)
    ratios = torch.tensor([r[3]/max(r[2], 1e-9) for r in rows])
    print(f"\n  cert/PGD tightness: mean {ratios.mean():.2f}x median {ratios.median():.2f}x")
    print("  cert VIOLATED must be 0: every output we certify <= delta is ACTUALLY <= delta")
    print("  under the strongest attack -> certified-trustworthy IS trustworthy. test")
    print("  VIOLATED > 0: testing trusts outputs that the worst case breaks (unfounded).")


if __name__ == "__main__":
    main()
