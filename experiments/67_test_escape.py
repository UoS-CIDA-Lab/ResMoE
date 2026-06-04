"""Experiment 67 — Testing gives false confidence; our certificate is sound (the
money experiment for the ICSE "verified compression gate" paper).

SCENARIO (offline MLOps/CI gate): a pipeline quantizes an MoE and must assert each
expert is behaviorally equivalent to the original within tolerance delta over its
OPERATING REGION (an L2 eps-ball around the calibration manifold). Two gates:
  - TESTING (current practice): max ||E - E_q||_inf over a finite SAMPLE of the region
      = clean calibration points + M Monte-Carlo perturbations in B2(h, eps).
      PASSES if sampled-max < delta. Can MISS in-region inputs (a TEST ESCAPE).
  - VERIFICATION (ours): the certified upper bound (exp 54) over the WHOLE ball.
      FLAGS if cert > delta. SOUND: if it certifies, no in-region input exceeds delta.
GROUND TRUTH worst case is lower-bounded by PGD over the ball. A TEST ESCAPE =
testing PASSES (sampled-max < delta) yet PGD finds an input with deviation > delta.
We show: testing has a nonzero escape rate; our gate has ZERO escapes (sound) and
few false alarms (tight, since cert ~ PGD).

CPU, OLMoE layer-0 cache. Run: python3 experiments/67_test_escape.py
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
MC_SAMPLES = 256        # Monte-Carlo "testing" budget per (expert,bit)
PGD_STEPS = 150
N_EXPERTS_EVAL = 40     # active experts to study


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


def mc_test(h0, W1, W2, W3, W1q, W2q, W3q, eps, m):
    """Max ||E - E_q||_inf over m random samples in B2(h0, eps) + the clean point."""
    h = h0.squeeze(0)
    devs = [(swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).abs().max().item()]
    for _ in range(m // 64 + 1):
        z = torch.randn(64, h.shape[0])
        z = z / z.norm(dim=1, keepdim=True) * (torch.rand(64, 1) ** (1/ h.shape[0])) * eps
        x = h.unsqueeze(0) + z
        d = (swiglu(x, W1, W2, W3) - swiglu(x, W1q, W2q, W3q)).abs().amax(1)
        devs.append(d.max().item())
    return max(devs)


def pgd_worst(h0, W1, W2, W3, W1q, W2q, W3q, eps, steps):
    """PGD lower bound on max ||E - E_q||_inf over B2(h0, eps)."""
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
    logits = H @ Wg.T
    topk = logits.topk(K, dim=-1).indices
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool); inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()][:N_EXPERTS_EVAL]
    print(f"  {len(active)} experts, eps_L2={EPS}, MC budget={MC_SAMPLES}", flush=True)

    rows = []   # (expert, bit, h-token, clean, mc, pgd, cert)
    for e in active:
        t = torch.nonzero(inset[:, e]).flatten()[0].item()
        h0 = H[t:t+1]
        s1, s2 = bigspec(W1[e]), bigspec(W2[e])
        for b in BITS:
            W1q, W2q, W3q = quant(W1[e], b), quant(W2[e], b), quant(W3[e], b)
            sv = (s1, s2, bigspec(W1q), bigspec(W2q), bigspec(W1[e]-W1q), bigspec(W2[e]-W2q))
            cert = cert_bound(W1[e], W2[e], W3[e], W1q, W2q, W3q, h0, EPS, Ls, Lpp, Lppp, sv)
            mc = mc_test(h0, W1[e], W2[e], W3[e], W1q, W2q, W3q, EPS, MC_SAMPLES)
            pgd = pgd_worst(h0, W1[e], W2[e], W3[e], W1q, W2q, W3q, EPS, PGD_STEPS)
            clean = (swiglu(h0, W1[e], W2[e], W3[e])
                     - swiglu(h0, W1q, W2q, W3q)).abs().max().item()
            rows.append((e, b, clean, mc, pgd, cert))
        print(f"  expert {e} done", flush=True)

    # gate analysis over a sweep of tolerances delta
    print(f"\n{'='*74}\n  Testing vs Verification gate (eps_L2={EPS}, {len(rows)} expert-bit cases)"
          f"\n{'='*74}")
    print(f"  delta   | test PASS | true-unsafe | TEST ESCAPES | cert FLAG | cert FALSE-alarm "
          f"| cert UNSOUND")
    print("  " + "-"*92)
    for delta in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
        test_pass = sum(mc < delta for _, _, _, mc, _, _ in rows)
        unsafe = sum(pgd > delta for _, _, _, _, pgd, _ in rows)
        escape = sum((mc < delta) and (pgd > delta) for _, _, _, mc, pgd, _ in rows)
        flag = sum(cert > delta for _, _, _, _, _, cert in rows)
        false_alarm = sum((cert > delta) and (pgd <= delta) for _, _, _, _, pgd, cert in rows)
        unsound = sum((cert <= delta) and (pgd > delta) for _, _, _, _, pgd, cert in rows)
        print(f"  {delta:>5.3f}   | {test_pass:>9d} | {unsafe:>11d} | {escape:>12d} | "
              f"{flag:>9d} | {false_alarm:>16d} | {unsound:>12d}")

    # tightness
    ratios = torch.tensor([cert / max(pgd, 1e-9) for _, _, _, _, pgd, cert in rows])
    print(f"\n  cert/PGD tightness: mean {ratios.mean():.2f}x median {ratios.median():.2f}x "
          f"(1.0 = exact). MC/PGD: "
          f"{torch.tensor([mc/max(pgd,1e-9) for _,_,_,mc,pgd,_ in rows]).mean():.2f}x "
          f"(<1 => testing UNDER-estimates the worst case).")
    print(f"\n  TEST ESCAPES = unsafe quantizations that testing SHIPS; cert UNSOUND must be 0")
    print("  (sound); cert FALSE-alarm small (tight). This is the SE value: a sound gate.")


if __name__ == "__main__":
    main()
