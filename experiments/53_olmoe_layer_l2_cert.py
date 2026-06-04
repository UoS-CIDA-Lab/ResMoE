"""Experiment 53 — LAYER-level L2 certified quant robustness for OLMoE SwiGLU.

Completes exp 52 (per-expert) into a full MoE-layer certificate. Quantization
leaves the router UNCHANGED, so original and quantized layers select the SAME
top-K with the SAME gates at every x:
    Y(x) - Y_q(x) = sum_{e in topK(x)} g_e(x) (E_e(x) - E_{e,q}(x)).
Since g_e >= 0 and sum_{e in topK} g_e <= 1 (softmax mass), this is a convex
combination, so
    ||Y(x)-Y_q(x)||_inf <= max_{e reachable in ball} ||E_e - E_{e,q}||_inf
                        <= max_{e reachable} (exp-52 per-expert L2 bound).
"reachable" = experts that can be in top-K for some x in the L2 eps-ball,
computed soundly from router logit intervals (l_e(x) in l_e(h0) +- eps||w_e||_2).

Reports, over the L2 eps-ball: layer bound vs layer L2-PGD (looseness), and the
reachable-set size. SOUND (>= PGD) and tight => a usable per-input layer-level
certified equivalence-robustness radius for quantized SwiGLU MoE.
Run: python3 experiments/53_olmoe_layer_l2_cert.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


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
    g2, = torch.autograd.grad(g1.sum(), z)
    return g1.abs().max().item() * 1.01, g2.abs().max().item() * 1.01


def quantize_per_channel(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / scale).clamp(-qmax - 1, qmax) * scale


def expert_l2_bound(W1, W2, W3, W1q, W2q, W3q, h0, eps, Ls, Lpp, s1, s2, s1q, s2q):
    """exp-52 per-expert L2 bound on ||E - E_q||_inf over the L2 eps-ball."""
    lo, hi = h0.squeeze(0) - eps, h0.squeeze(0) + eps      # superset of L2 ball
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi))
    Bvq = amax(*affine_box(W2q, lo, hi))
    def hspec(W3_, s1_, s2_, Bv_):
        maxc = Lpp * (W3_.abs() * Bv_.unsqueeze(0)).amax(1)
        maxe = Ls * W3_.abs().amax(1)
        return s1_ * s1_ * maxc + 2 * s1_ * s2_ * maxe
    HE = hspec(W3, s1, s2, Bv)
    HEq = hspec(W3q, s1q, s2q, Bvq)
    h = h0.squeeze(0)
    def J(A, B, C):
        u = A @ h; v = B @ h
        return (C * (silu_grad(u) * v).unsqueeze(0)) @ A + (C * F.silu(u).unsqueeze(0)) @ B
    JD2 = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0, W1, W2, W3) - swiglu(h0, W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps * JD2 + 0.5 * eps * eps * (HE + HEq)).max().item()


def reachable_experts(h0, Wg, eps, K):
    """Experts that can be in top-K within the L2 eps-ball (sound)."""
    l = (h0 @ Wg.T).squeeze(0)                    # [N]
    wn = Wg.norm(dim=1)                            # ||w_e||_2
    lb = l - eps * wn
    ub = l + eps * wn
    tau = lb.topk(K).values.min()                  # K-th largest lower bound
    return torch.nonzero(ub >= tau).flatten().tolist()


@torch.no_grad()
def layer_forward(x, Wg, W1, W2, W3, K):
    logits = x @ Wg.T
    probs = F.softmax(logits, dim=-1)
    topv, topi = probs.topk(K, dim=-1)
    y = torch.zeros_like(x)
    for s in range(K):
        idx = topi[:, s]
        g = topv[:, s:s + 1]
        for e in idx.unique().tolist():
            m = idx == e
            y[m] += g[m] * swiglu(x[m], W1[e], W2[e], W3[e])
    return y


def layer_l2_pgd(h0, Wg, W1, W2, W3, W1q, W2q, W3q, K, eps, n_steps=150):
    step = eps / 15 if eps > 0 else 0.0
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps if eps > 0 else 1):
        hg = h.clone().requires_grad_(True)
        # differentiable gated forward (reuse routing from current h)
        logits = hg @ Wg.T
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(K, dim=-1)
        yo = torch.zeros_like(hg); yq = torch.zeros_like(hg)
        for s in range(K):
            e = int(topi[0, s]); g = topv[:, s:s + 1]
            yo = yo + g * swiglu(hg, W1[e], W2[e], W3[e])
            yq = yq + g * swiglu(hg, W1q[e], W2q[e], W3q[e])
        loss = (yo - yq).abs().max()
        loss.backward()
        with torch.no_grad():
            grad = hg.grad
            h = h + step * grad / grad.norm().clamp(min=1e-12)
            delta = h - h0
            dn = delta.norm()
            if dn > eps:
                h = h0 + delta * (eps / dn)
            yo2 = layer_forward(h, Wg, W1, W2, W3, K)
            yq2 = layer_forward(h, Wg, W1q, W2q, W3q, K)
            best = max(best, (yo2 - yq2).abs().max().item())
    return best


def main():
    print("Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float()
    H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    Ls, Lpp = silu_consts()
    torch.manual_seed(0)
    pool = torch.randperm(H.shape[0])[:15]

    for bits in (8, 4):
        W1q = torch.stack([quantize_per_channel(W1[i], bits) for i in range(N)])
        W2q = torch.stack([quantize_per_channel(W2[i], bits) for i in range(N)])
        W3q = torch.stack([quantize_per_channel(W3[i], bits) for i in range(N)])
        sn = lambda W: torch.linalg.matrix_norm(W, ord=2)
        s1 = [sn(W1[i]).item() for i in range(N)]
        s2 = [sn(W2[i]).item() for i in range(N)]
        s1q = [sn(W1q[i]).item() for i in range(N)]
        s2q = [sn(W2q[i]).item() for i in range(N)]
        print(f"\n{'='*70}\n  {bits}-bit: LAYER L2 cert vs layer L2-PGD  (top-{K} of {N})"
              f"\n{'='*70}")
        print(f"  {'eps':>6s} {'layer-PGD':>11s} {'layer-bound':>12s} {'×':>7s} "
              f"{'reach':>6s} {'sound?':>7s}")
        print("  " + "-" * 52)
        for eps in (0.0, 0.01, 0.05, 0.1):
            E, B, R = [], [], []
            for bi in pool.tolist():
                h = H[bi:bi + 1]
                reach = reachable_experts(h, Wg, eps, K)
                R.append(len(reach))
                bnd = max(expert_l2_bound(W1[e], W2[e], W3[e], W1q[e], W2q[e],
                                          W3q[e], h, eps, Ls, Lpp,
                                          s1[e], s2[e], s1q[e], s2q[e])
                          for e in reach)
                B.append(bnd)
                E.append(layer_l2_pgd(h, Wg, W1, W2, W3, W1q, W2q, W3q, K, eps))
            me = sorted(E)[len(E) // 2]; mb = sorted(B)[len(B) // 2]
            mr = sum(R) / len(R)
            sound = "yes" if all(bb >= ee - 1e-6 for bb, ee in zip(B, E)) else "NO!"
            print(f"  {eps:>6.3f} {me:>11.4f} {mb:>12.4f} {mb/max(me,1e-9):>6.1f}× "
                  f"{mr:>6.1f} {sound:>7s}")

    print(f"\n{'='*70}")
    print("  Layer bound = max over reachable experts of the per-expert L2 bound")
    print("  (convex-combination of gated quant errors). Sound + tight at small")
    print("  eps => per-input LAYER-level L2 certified robustness for quant SwiGLU.")


if __name__ == "__main__":
    main()
