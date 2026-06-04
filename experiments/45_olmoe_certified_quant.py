"""Experiment 45 — Certified QUANTIZATION (extension ②), routing-orthogonal.

Structurally identical to certified aliasing (exp 44): keep the router, certify
||E(x) - E'(x)|| over an eps-ball, but here E' = the QUANTIZED expert (weights
rounded to b bits) rather than a twin. Advantages of quantization as the
operator: (a) applies to ALL experts uniformly (no aliasing-budget problem),
(b) reduces BOTH memory AND compute (low-bit GEMM) — unlike pruning/aliasing
which are memory-only.

Question: since E and E_quant differ only by a tiny weight perturbation
||dW||_inf <= step/2, is the SwiGLU difference bound TIGHTER than the
expert-vs-expert aliasing bound (exp 44: 21.9x@eps0.01)? Or does McCormick's
per-term over-relaxation (before the W3 cancellation) stay loose regardless?

Per-channel symmetric quant. Reports empirical quant deviation (near-lossless?)
and the certified per-input eps-ball bound vs PGD (looseness), per bit-width.

Run: python3 experiments/45_olmoe_certified_quant.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.swiglu_bounds import swiglu_diff_bound_box

CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def quantize_per_channel(W, bits):
    """Symmetric per-output-channel (per-row) quantization."""
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / scale).clamp(-qmax - 1, qmax) * scale


def pgd_pair_diff(Wi, Wj, h0, eps, n_steps=100):
    (W1i, W2i, W3i), (W1j, W2j, W3j) = Wi, Wj
    step = eps / 10 if eps > 0 else 0.0
    h = h0.clone().detach()
    best = 0.0
    for _ in range(max(1, n_steps if eps > 0 else 1)):
        hg = h.clone().requires_grad_(True)
        d = (swiglu(hg, W1i, W2i, W3i) - swiglu(hg, W1j, W2j, W3j)
             ).abs().max(-1).values.sum()
        d.backward()
        with torch.no_grad():
            if eps > 0:
                h = torch.max(torch.min(h + step * hg.grad.sign(), h0 + eps),
                              h0 - eps)
            cur = (swiglu(h, W1i, W2i, W3i) - swiglu(h, W1j, W2j, W3j)
                   ).abs().max(-1).values.max().item()
            best = max(best, cur)
    return best


def main():
    print("Loading OLMoE layer-0 cache...")
    c = torch.load(CACHE, weights_only=True)
    W1, W2, W3 = c["experts_W1"].float(), c["experts_W2"].float(), c["experts_W3"].float()
    Wg = c["router_weight"].float()
    H = c["H"].float()
    N, K = c["n_experts"], c["top_k"]
    print(f"  N={N}, K={K}, H={tuple(H.shape)}, ||h|| mean {H.norm(dim=-1).mean():.1f}")

    logits = H @ Wg.T
    _, topk_idx = logits.topk(K, dim=-1)
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool)
    inset.scatter_(1, topk_idx, True)

    for bits in (8, 4):
        # quantize all experts
        W1q = torch.stack([quantize_per_channel(W1[i], bits) for i in range(N)])
        W2q = torch.stack([quantize_per_channel(W2[i], bits) for i in range(N)])
        W3q = torch.stack([quantize_per_channel(W3[i], bits) for i in range(N)])

        # empirical clean quant deviation per expert over its routed inputs
        clean = []
        for i in range(N):
            Hi = H[inset[:, i]]
            if Hi.shape[0] == 0:
                continue
            with torch.no_grad():
                d = (swiglu(Hi, W1[i], W2[i], W3[i])
                     - swiglu(Hi, W1q[i], W2q[i], W3q[i])).abs().max(-1).values.max()
            clean.append(d.item())
        out_scale = swiglu(H, W1[0], W2[0], W3[0]).abs().max(-1).values.mean().item()

        print(f"\n{'='*64}\n  {bits}-bit per-channel quantization "
              f"(expert out |y|_inf ~ {out_scale:.2f})\n{'='*64}")
        print(f"  empirical clean quant dev (routed): median "
              f"{sorted(clean)[len(clean)//2]:.4f}, max {max(clean):.4f}")

        # per-input eps-ball certified bound vs PGD (sample inputs, use top-1 expert)
        torch.manual_seed(0)
        pool = torch.randperm(H.shape[0])[:20]
        print(f"  {'eps':>7s} {'med cert':>10s} {'med emp':>10s} {'looseness':>10s}")
        print("  " + "-" * 42)
        for eps in (0.0, 0.01, 0.05):
            certs, emps = [], []
            for b in pool.tolist():
                h = H[b:b + 1]
                i = int(logits[b].argmax())
                cert = swiglu_diff_bound_box(
                    W1[i], W2[i], W3[i], W1q[i], W2q[i], W3q[i],
                    (h - eps).squeeze(0), (h + eps).squeeze(0))
                emp = pgd_pair_diff((W1[i], W2[i], W3[i]),
                                    (W1q[i], W2q[i], W3q[i]), h, eps)
                certs.append(float(cert)); emps.append(emp)
            mc = sorted(certs)[len(certs) // 2]
            me = sorted(emps)[len(emps) // 2]
            print(f"  {eps:>7.3f} {mc:>10.4f} {me:>10.4f} {mc/max(me,1e-6):>9.1f}x")

    print(f"\n{'='*64}")
    print("  Compare looseness to exp44 aliasing (21.9x@eps0.01). Does the tiny")
    print("  quant dW give a tighter SwiGLU bound, or does McCormick stay loose?")
    print("  eps=0 row = pure quant-error bound (degenerate box).")


if __name__ == "__main__":
    main()
