"""Experiment 63 — Certified ROUTING EQUIVALENCE under routing-orthogonal quant.

THESIS PIVOT. Per-layer OUTPUT certificates do not compose end-to-end (Lipschitz
amplification -> vacuous). But ROUTING is a DISCRETE decision with a margin: if at
every layer the router's top-K margin exceeds the quantization-induced hidden-state
drift, the top-K SET is preserved EXACTLY, and discrete equality composes with NO
amplification. So we can give an end-to-end statement that output-certification
cannot: the certified-quantized MoE routes every token through the SAME experts as
the fp16 model -> a structurally-faithful execution of the original compute graph.

Router is linear (h -> W_g h), so the routing-preservation certificate is TIGHT:
at fp16 hidden state h_l, top-K is invariant under any perturbation of L2-radius r iff
    margin_l(h) = min_{e in S}(logit_e - r||w_e||) - max_{e not in S}(logit_e + r||w_e||) > 0.
We take r = the ACTUAL per-token drift d_l = ||h_l^q - h_l||_2 (measured by running
both models). If margin_l(h) > 0 for all l on a token, that token is CERTIFIED to be
routed identically by the quantized model (sound: ||h^q - h|| <= d_l <= r).

Outputs: (1) empirical per-layer top-K agreement fp16-vs-quant; (2) certified
routing-preservation rate vs the measured drift; (3) the certified routing RADIUS
(max r with margin>0) distribution vs the drift scale. GPU, 75% cap.
Run AFTER exp 60 v2. python3 experiments/63_routing_equivalence.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
EPS = 0.05
DELTA_LYR = 0.01
BITS = [16, 8, 6, 4, 3, 2]
N_CALIB = 256
N_EVAL = 256       # tokens to evaluate routing equivalence on (held-out)


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
def expert_bound(W1, W2, W3, W1q, W2q, W3q, h0, eps, sv, Ls, Lpp, Lppp):
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
def layer_budget(block, Hc, dev, Ls, Lpp, Lppp):
    K, N = block.top_k, block.num_experts
    Wg = block.gate.weight.float()
    logits = Hc @ Wg.T
    topv, topk = logits.topk(K, dim=-1)
    gates = torch.softmax(topv, dim=-1)
    T = Hc.shape[0]
    inset = torch.zeros(T, N, dtype=torch.bool, device=dev); inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist() for e in active}
    cert = {e: {16: 0.0} for e in active}
    for e in active:
        W1 = block.experts[e].gate_proj.weight.float()
        W2 = block.experts[e].up_proj.weight.float()
        W3 = block.experts[e].down_proj.weight.float()
        s1, s2 = spec(W1), spec(W2)
        for b in BITS:
            if b >= 16:
                continue
            W1q, W2q, W3q = quant(W1, b), quant(W2, b), quant(W3, b)
            sv = (s1, s2, spec(W1q), spec(W2q), spec(W1-W1q), spec(W2-W2q))
            cert[e][b] = max(expert_bound(W1, W2, W3, W1q, W2q, W3q, Hc[t], EPS, sv,
                                          Ls, Lpp, Lppp) for t in routed[e])
    bits = {e: 2 for e in active}
    nxt = {2: 3, 3: 4, 4: 6, 6: 8, 8: 16, 16: 16}
    for _ in range(20000):
        worst_t, worst_s = -1, DELTA_LYR
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
    return {e: bits.get(e, 2) for e in range(N)}


@torch.no_grad()
def collect_hidden(model, ids, dev, nL, n):
    caps = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
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
    return {li: torch.cat(caps[li]) for li in range(nL)}


@torch.no_grad()
def quantize_model(model, bits_per_layer):
    for li, layer in enumerate(model.model.layers):
        bk = bits_per_layer[li]
        for e, ex in enumerate(layer.mlp.experts):
            b = bk[e]
            if b >= 16:
                continue
            for proj in ("gate_proj", "up_proj", "down_proj"):
                w = getattr(ex, proj).weight
                getattr(ex, proj).weight.copy_(quant(w.float(), b).to(w.dtype))


@torch.no_grad()
def topk_sets(block, Hc):
    logits = Hc @ block.gate.weight.float().T
    _, topk = logits.topk(block.top_k, dim=-1)
    return topk, logits


@torch.no_grad()
def cert_radius(logits, topk, rownorm, K, N):
    """Per-token max L2 radius r with top-K SET provably invariant (router linear)."""
    T = logits.shape[0]
    inset = torch.zeros(T, N, dtype=torch.bool, device=logits.device)
    inset.scatter_(1, topk, True)
    # invariant iff for the binding pair, gap > r*(||w_sel|| + ||w_uns||). The largest
    # r keeping ALL selected above ALL unselected: r* = min over (s in S, u not in S)
    # of (logit_s - logit_u)/(||w_s|| + ||w_u||). Compute via the binding (smallest) pair.
    rn = rownorm
    sel_lo_logit = logits.masked_fill(~inset, float("inf"))   # selected logits
    uns_hi_logit = logits.masked_fill(inset, float("-inf"))   # unselected logits
    # For each token: r* = min_{s,u} (l_s - l_u)/(||w_s||+||w_u||).
    radii = torch.empty(T, device=logits.device)
    for t in range(T):
        S = inset[t]
        ls = logits[t][S]; lu = logits[t][~S]
        ws = rn[S]; wu = rn[~S]
        # pairwise (l_s - l_u)/(w_s + w_u), min over all pairs
        num = ls[:, None] - lu[None, :]
        den = ws[:, None] + wu[None, :]
        radii[t] = (num / den).min()
    return radii


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calib_ids = tok("\n\n".join(t for t in train["text"] if t.strip()),
                    return_tensors="pt").input_ids[0]
    eval_ids = tok("\n\n".join(t for t in test["text"] if t.strip()),
                   return_tensors="pt").input_ids[0]

    print("Loading model + certified budget...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    nL = len(model.model.layers)
    Ls, Lpp, Lppp = silu_consts(dev)
    calib = collect_hidden(model, calib_ids, dev, nL, N_CALIB)
    budgets = {li: layer_budget(model.model.layers[li].mlp, calib[li], dev, Ls, Lpp, Lppp)
               for li in range(nL)}
    for li in range(nL):
        print(f"  layer {li} budget done", flush=True)

    # fp16 hidden states + routing on held-out eval tokens
    H_fp = collect_hidden(model, eval_ids, dev, nL, N_EVAL)
    route_fp = {li: topk_sets(model.model.layers[li].mlp, H_fp[li]) for li in range(nL)}
    rownorm = {li: model.model.layers[li].mlp.gate.weight.float().norm(dim=1)
               for li in range(nL)}
    K = model.model.layers[0].mlp.top_k
    N = model.model.layers[0].mlp.num_experts

    # quantize, recompute hidden states + routing on the SAME eval tokens
    quantize_model(model, budgets)
    H_q = collect_hidden(model, eval_ids, dev, nL, N_EVAL)
    route_q = {li: topk_sets(model.model.layers[li].mlp, H_q[li]) for li in range(nL)}

    print(f"\n{'='*76}\n  Certified ROUTING EQUIVALENCE (certified RA budget, delta={DELTA_LYR})"
          f"\n{'='*76}")
    print(f"  {'layer':>5s} {'drift d_l (mean/max)':>22s} {'emp top-K agree':>16s} "
          f"{'cert-preserved':>15s} {'cert radius>drift':>17s}")
    tot_emp = tot_cert = tot_n = 0
    for li in range(nL):
        topk_fp, logits_fp = route_fp[li]
        topk_q, _ = route_q[li]
        T = min(topk_fp.shape[0], topk_q.shape[0])
        # empirical: top-K SET agreement per token
        sfp = [set(topk_fp[t].tolist()) for t in range(T)]
        sq = [set(topk_q[t].tolist()) for t in range(T)]
        emp = sum(sfp[t] == sq[t] for t in range(T)) / T
        # drift per token (align lengths)
        d = (H_q[li][:T] - H_fp[li][:T]).norm(dim=1)
        # certified routing radius at fp16 state, vs measured drift
        radii = cert_radius(logits_fp[:T], topk_fp[:T], rownorm[li], K, N)
        cert_ok = (radii > d).float().mean().item()
        print(f"  {li:>5d} {d.mean().item():>10.4f}/{d.max().item():<10.4f} "
              f"{emp*100:>15.1f}% {cert_ok*100:>14.1f}% {radii.mean().item():>10.4f}",
              flush=True)
        tot_emp += emp*T; tot_cert += cert_ok*T; tot_n += T
    print("  " + "-"*72)
    print(f"  ALL LAYERS: empirical top-K agreement {tot_emp/tot_n*100:.1f}%, "
          f"certified-preserved {tot_cert/tot_n*100:.1f}%")
    print(f"\n  cert-preserved = fraction of (token,layer) where the router's top-K SET")
    print("  is PROVABLY invariant under the measured quant drift d_l (router linear,")
    print("  tight). Where cert holds at every layer, the token is routed through the")
    print("  EXACT same experts as fp16 -> structurally-faithful certified execution.")


if __name__ == "__main__":
    main()
