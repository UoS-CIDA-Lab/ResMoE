"""Experiment 71 — The decisive test of the worst-case advantage:
CERTIFIED (region worst-case) allocation  vs  EMPIRICAL (calib-deviation) allocation,
matched average bits, evaluated UNDER DISTRIBUTION SHIFT.

Motivation (why exp 69 was inconclusive): exp 69 compared certified-selective vs
UNIFORM, on PGD around a single calib token, evaluated ON the calib region. That
measures the wrong thing. The honest worst-case advantage is GENERALIZATION:
empirical mixed-precision allocates bits by the deviation it OBSERVED on the
calibration set, so it under-protects experts that look benign on calib but are
intrinsically high-curvature; the certified bound (driven by spectral curvature /
||dW||, not by where calib points happened to land) protects them. The difference
must therefore show up on inputs the allocation did NOT see -- a SHIFTED domain.

Protocol (per layer):
  calib  = prose (WikiText) tokens   -> both allocators built here.
  shift  = code (HumanEval) + math (GSM8K) tokens -> both evaluated here.
  Empirical allocator (standard practice): bits[e] = coarsest b s.t.
     max_{h in prose-routed(e)} ||E_e(h) - E_{e,q}(h)||_inf <= tau.   (no eps-ball)
  Certified allocator (ours): bits[e] = coarsest b s.t.
     max_{h in prose-routed(e)} cert_eps(e,h,b) <= delta.             (eps-ball, exp54)
  Sweep tau / delta so the two reach MATCHED average bits, then compare on shift:
     (a) max-over-experts clean deviation ||E - E_q||_inf  (layer worst expert),
     (b) gate-weighted per-token layer deviation  sum_e g_e ||E_e - E_{e,q}||_inf
         (worst token & mean), the actual layer output error under shift.
We also report in-domain (prose held-out) so the reader sees the trade.

If certified < empirical on shift at matched bits -> the worst-case advantage is
real and measurable. If ~equal -> MoE compression is intrinsically robust (routing
redundancy) and the guarantee buys nothing measurable; a clean thesis verdict either way.

GPU, live OLMoE. Run: python3 experiments/71_cert_vs_empirical_shift.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
EPS = 0.05
BITS = [8, 6, 4, 3, 2]
N_CALIB = 512
N_SHIFT = 512
LAYERS = [0, 7]
TARGET_BITS = [6.0, 5.0, 4.0, 3.0]


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def silu_grad(z):
    s = torch.sigmoid(z)
    return s + z * s * (1 - s)


def affine_box(W, lo, hi):
    c = (lo + hi) / 2; r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_consts(dev):
    z = torch.linspace(-30, 30, 400001, device=dev).requires_grad_(True)
    g1, = torch.autograd.grad(F.silu(z).sum(), z, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    return (g1.abs().max().item()*1.01, g2.abs().max().item()*1.01,
            g3.abs().max().item()*1.01)


def quant(W, bits):
    if bits >= 16:
        return W
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def spec(W):
    return torch.linalg.matrix_norm(W.float(), ord=2).item()


@torch.no_grad()
def cert_eps(W1, W2, W3, W1q, W2q, W3q, h0, eps, sv, Ls, Lpp, Lppp):
    """exp-54 per-input L2 eps-ball certified bound on ||E-E_q||_inf at h0 (1D)."""
    s1, s2, s1q, s2q, sd1, sd2 = sv
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    lo, hi = h0 - eps, h0 + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi)); Bvq = amax(*affine_box(W2q, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi)); Bd2 = amax(*affine_box(d2, lo, hi))
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)
    De = Ls * W3.abs().amax(1)
    dDc = (W3.abs()*(Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0)
           + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs()*(Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    HD = Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)
    def J(A, B, C):
        u = A @ h0; v = B @ h0
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JD = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0.unsqueeze(0), W1, W2, W3)
           - swiglu(h0.unsqueeze(0), W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps*JD + 0.5*eps*eps*HD).max().item()


@torch.no_grad()
def collect(model, ids, dev, layers, n):
    caps = {li: [] for li in layers}
    hs = []
    for li in layers:
        def mk(li):
            def hk(_m, a):
                caps[li].append(a[0].detach().reshape(-1, a[0].shape[-1]).float())
            return hk
        hs.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    try:
        model(ids[:n].unsqueeze(0).to(dev))
    finally:
        for h in hs:
            h.remove()
    return {li: torch.cat(caps[li]) for li in layers}


def task_streams(tok):
    from datasets import load_dataset
    out = {}
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    out["prose"] = "\n\n".join(t for t in wt["text"] if t.strip())
    he = load_dataset("openai_humaneval", split="test")
    out["code"] = "\n\n".join(he["prompt"])
    gs = load_dataset("gsm8k", "main", split="test")
    out["math"] = "\n\n".join(gs["question"][:400])
    return {k: tok(v, return_tensors="pt").input_ids[0] for k, v in out.items()}


@torch.no_grad()
def routed_inputs(block, H):
    """Return (active list, {e: idx tensor}, gates[T,N] dense, topk[T,K])."""
    K, N = block.top_k, block.num_experts
    Wg = block.gate.weight.float()
    logits = H @ Wg.T
    topv, topk = logits.topk(K, dim=-1)
    gates_sel = torch.softmax(topv, dim=-1)
    T = H.shape[0]
    inset = torch.zeros(T, N, dtype=torch.bool, device=H.device)
    inset.scatter_(1, topk, True)
    gates = torch.zeros(T, N, device=H.device)
    gates.scatter_(1, topk, gates_sel)
    active = [e for e in range(N) if inset[:, e].any()]
    routed = {e: torch.nonzero(inset[:, e]).flatten() for e in active}
    return active, routed, gates, topk


@torch.no_grad()
def per_input_dev(block, e, H, idx, bits):
    """||E_e(h) - E_{e,q}(h)||_inf for each routed input h (vector over idx)."""
    if len(idx) == 0:
        return torch.zeros(0, device=H.device)
    W1 = block.experts[e].gate_proj.weight.float()
    W2 = block.experts[e].up_proj.weight.float()
    W3 = block.experts[e].down_proj.weight.float()
    W1q, W2q, W3q = quant(W1, bits), quant(W2, bits), quant(W3, bits)
    Hi = H[idx]
    d = (swiglu(Hi, W1, W2, W3) - swiglu(Hi, W1q, W2q, W3q)).abs().amax(1)
    return d


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.80, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    streams = task_streams(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    Ls, Lpp, Lppp = silu_consts(dev)

    Hp = collect(model, streams["prose"], dev, LAYERS, N_CALIB)
    Hc = collect(model, streams["code"], dev, LAYERS, N_SHIFT)
    Hm = collect(model, streams["math"], dev, LAYERS, N_SHIFT)

    for li in LAYERS:
        block = model.model.layers[li].mlp
        N, K = block.num_experts, block.top_k
        Hpl = Hp[li]
        Hshift = torch.cat([Hc[li], Hm[li]])   # combined shift domain
        active, routed_p, _, _ = routed_inputs(block, Hpl)
        _, routed_s, gates_s, topk_s = routed_inputs(block, Hshift)
        print(f"\n{'#'*80}\n  LAYER {li}: {len(active)} active experts (prose calib), "
              f"shift={Hshift.shape[0]} tok (code+math)\n{'#'*80}", flush=True)

        # per-expert per-bit: certified (eps-ball, prose region) and empirical (clean, prose)
        cert = {e: {16: 0.0} for e in active}
        emp = {e: {16: 0.0} for e in active}
        for e in active:
            W1 = block.experts[e].gate_proj.weight.float()
            W2 = block.experts[e].up_proj.weight.float()
            W3 = block.experts[e].down_proj.weight.float()
            s1, s2 = spec(W1), spec(W2)
            idx = routed_p[e]
            for b in BITS:
                W1q, W2q, W3q = quant(W1, b), quant(W2, b), quant(W3, b)
                sv = (s1, s2, spec(W1q), spec(W2q), spec(W1-W1q), spec(W2-W2q))
                cert[e][b] = max(cert_eps(W1, W2, W3, W1q, W2q, W3q, Hpl[t], EPS,
                                          sv, Ls, Lpp, Lppp) for t in idx.tolist())
                d = (swiglu(Hpl[idx], W1, W2, W3)
                     - swiglu(Hpl[idx], W1q, W2q, W3q)).abs().amax(1)
                emp[e][b] = d.max().item()
        print("  per-expert cert & empirical-calib bounds done", flush=True)

        def alloc(scoremap, thr):
            out = {}
            for e in active:
                feas = [b for b in BITS if scoremap[e][b] <= thr]
                out[e] = min(feas) if feas else 16
            return out

        def avg_bits(bits):
            return sum(bits.values()) / len(active)

        def match(scoremap, target, grid):
            best, bestgap = None, 1e9
            for thr in grid:
                a = avg_bits(alloc(scoremap, thr))
                if abs(a - target) < bestgap:
                    bestgap, best = abs(a - target), thr
            return best

        grid = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03,
                0.05, 0.08, 0.12, 0.2, 0.3, 0.5]

        # precompute per-expert per-bit SHIFT deviation (clean) and per-token contrib
        # store max-over-shift-routed dev and the full per-token dev for gate weighting
        shift_dev_max = {e: {16: 0.0} for e in active}
        shift_dev_vec = {e: {} for e in active}     # b -> dev per shift-routed input
        for e in active:
            idx = routed_s.get(e, torch.zeros(0, dtype=torch.long, device=dev))
            for b in BITS + [16]:
                d = per_input_dev(block, e, Hshift, idx, b)
                shift_dev_max[e][b] = (d.max().item() if len(d) else 0.0)
                shift_dev_vec[e][b] = d

        def shift_metrics(bits):
            """max-over-expert clean dev + gate-weighted per-token layer dev on shift."""
            # (a) worst expert
            we = max((shift_dev_max[e][bits[e]] for e in active), default=0.0)
            # (b) gate-weighted per-token: build [T] sum_e g_e dev_e at assigned bits
            T = Hshift.shape[0]
            tok_dev = torch.zeros(T, device=dev)
            for e in active:
                idx = routed_s.get(e, None)
                if idx is None or len(idx) == 0:
                    continue
                tok_dev[idx] += gates_s[idx, e] * shift_dev_vec[e][bits[e]]
            return we, tok_dev.max().item(), tok_dev.mean().item()

        print(f"\n  {'target':>6s} | {'method':9s} | {'avgbits':>7s} | "
              f"{'shift max-expert':>16s} | {'shift gw-worst':>14s} | {'shift gw-mean':>13s} "
              f"| {'prose max-exp':>13s}")
        print("  " + "-"*100)
        for tb in TARGET_BITS:
            tc = match(cert, tb, grid); te = match(emp, tb, grid)
            bc, be = alloc(cert, tc), alloc(emp, te)
            wc, gwc, gmc = shift_metrics(bc)
            we_, gwe, gme = shift_metrics(be)
            # in-domain prose worst expert for context
            pc = max(emp[e][bc[e]] for e in active)
            pe = max(emp[e][be[e]] for e in active)
            print(f"  {tb:>6.1f} | CERT (d={tc:<5g}) | {avg_bits(bc):>7.2f} | "
                  f"{wc:>16.4f} | {gwc:>14.4f} | {gmc:>13.5f} | {pc:>13.4f}", flush=True)
            print(f"  {tb:>6.1f} | EMP  (t={te:<5g}) | {avg_bits(be):>7.2f} | "
                  f"{we_:>16.4f} | {gwe:>14.4f} | {gme:>13.5f} | {pe:>13.4f}", flush=True)
            ratio = we_ / max(wc, 1e-9)
            print(f"         -> shift max-expert  EMP/CERT = {ratio:.2f}x   "
                  f"(>1 means certified wins under shift)\n")

    print("\nDONE. If EMP/CERT > 1 on shift while avg-bits matched -> certified "
          "allocation generalizes better (the worst-case advantage is real).")


if __name__ == "__main__":
    main()
