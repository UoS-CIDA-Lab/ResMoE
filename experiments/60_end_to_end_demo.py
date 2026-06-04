"""Experiment 60 — End-to-end application demo: certified mixed-precision OLMoE.

Applies the per-layer router-aware CERTIFIED quantization budget to ALL 16 layers
of the live OLMoE-1B-7B, builds the real mixed-precision model, and measures
actual memory + quality (WikiText PPL, HellaSwag) vs fp16 and vs uniform 4-bit at
matched memory. Small calibration/eval for speed.

HONEST scope: each layer's budget carries a per-layer L2 certificate (D_e bound);
end-to-end quality is measured EMPIRICALLY (per-layer certs do not yet compose to
a model-level certificate -- see Discussion). The demo shows the certified budget
yields a deployable model that runs and retains quality.

GPU capped at 75%. Run: python3 experiments/60_end_to_end_demo.py
"""
from __future__ import annotations

import sys
import pathlib
import math
import gc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
EPS = 0.05
DELTA_LYR = 0.01
BITS = [16, 8, 6, 4, 3, 2]
N_CALIB = 256   # uncapped D_e over all routed inputs is O(N_CALIB); 256 keeps
PPL_WINDOWS = 8
HS_N = 200


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
    return (g1.abs().max().item()*1.01, g2.abs().max().item()*1.01, g3.abs().max().item()*1.01)


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
    dDc = (W3.abs()*(Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0) + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs()*(Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    HD = Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)
    def J(A, B, C):
        u = A @ h0; v = B @ h0
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JD = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0.unsqueeze(0), W1, W2, W3) - swiglu(h0.unsqueeze(0), W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps*JD + 0.5*eps*eps*HD).max().item()


@torch.no_grad()
def layer_budget(block, Hc, dev, Ls, Lpp, Lppp):
    """Return {expert: bits} router-aware certified budget for one MoE layer."""
    K, N = block.top_k, block.num_experts
    Wg = block.gate.weight.float()
    logits = Hc @ Wg.T
    topv, topk = logits.topk(K, dim=-1)
    gates = torch.softmax(topv, dim=-1)
    T = Hc.shape[0]
    inset = torch.zeros(T, N, dtype=torch.bool, device=dev); inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    # NO cap: D_e must bound ALL inputs the expert serves for a sound per-layer cert.
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
            cert[e][b] = max(expert_bound(W1, W2, W3, W1q, W2q, W3q, Hc[t], EPS, sv, Ls, Lpp, Lppp)
                             for t in routed[e])
    # router-aware greedy on per-token constraint sum_{topK} g*D <= DELTA_LYR
    bits = {e: 2 for e in active}
    nxt = {2: 3, 3: 4, 4: 6, 6: 8, 8: 16, 16: 16}
    for _ in range(20000):
        worst_t, worst_s = -1, DELTA_LYR
        for t in range(T):
            s = sum(gates[t, j].item() * cert[topk[t, j].item()][bits[topk[t, j].item()]]
                    for j in range(K))
            if s > worst_s:
                worst_s, worst_t = s, t
        if worst_t < 0:
            break
        cand = [(gates[worst_t, j].item()*cert[topk[worst_t, j].item()][bits[topk[worst_t, j].item()]],
                 topk[worst_t, j].item()) for j in range(K) if bits[topk[worst_t, j].item()] != 16]
        if not cand:
            break
        e = max(cand)[1]
        bits[e] = nxt[bits[e]]
    full = {e: bits.get(e, 2) for e in range(N)}   # inactive experts -> 2-bit (free)
    return full


def load_model(dev):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    return m.to(dev).eval()


@torch.no_grad()
def collect_calib(model, ids, dev, nL, n):
    caps = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        def mk(li):
            def hk(_m, a): caps[li].append(a[0].detach().reshape(-1, a[0].shape[-1]).float())
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
def compute_ppl(model, ids, dev, ctx=1024, nwin=PPL_WINDOWS):
    nll = ntok = 0
    for i in range(0, ids.shape[0]-1, ctx):
        inp = ids[i:i+ctx].unsqueeze(0).to(dev)
        tgt = inp.clone(); tgt[:, :-1] = inp[:, 1:]; tgt[:, -1] = -100
        lg = model(inp).logits.float()
        sl = lg[:, :-1].reshape(-1, lg.size(-1)); st = tgt[:, :-1].reshape(-1)
        nll += F.cross_entropy(sl, st, reduction="sum").item(); ntok += (st != -100).sum().item()
        if i//ctx+1 >= nwin:
            break
    return math.exp(nll/max(ntok, 1))


@torch.no_grad()
def hellaswag(model, tok, ex, dev):
    cor = 0
    for ctx, conts, gold in ex:
        best, bl = -1, -1e30
        for ci, cont in enumerate(conts):
            cids = tok(ctx, add_special_tokens=True).input_ids
            kids = tok(cont, add_special_tokens=False).input_ids or [tok.eos_token_id]
            seq = torch.tensor(cids+kids, device=dev).unsqueeze(0)
            lp = F.log_softmax(model(seq).logits.float()[0], -1)
            s = sum(lp[len(cids)+k-1, kids[k]].item() for k in range(len(kids)))/len(kids)
            if s > bl:
                bl, best = s, ci
        cor += int(best == gold)
    return cor/len(ex)


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_ids = tok("\n\n".join(t for t in test["text"] if t.strip()), return_tensors="pt").input_ids[0]
    calib_ids = tok("\n\n".join(t for t in train["text"] if t.strip()), return_tensors="pt").input_ids[0]
    hs = load_dataset("hellaswag", split="validation")
    hex = [((hs[i]["ctx_a"]+" "+hs[i]["ctx_b"]).strip(), [" "+e for e in hs[i]["endings"]], int(hs[i]["label"]))
           for i in range(HS_N)]

    print("Loading model; collecting calib + computing per-layer budgets...", flush=True)
    model = load_model(dev)
    nL = len(model.model.layers)
    Ls, Lpp, Lppp = silu_consts(dev)
    calib = collect_calib(model, calib_ids, dev, nL, N_CALIB)
    ppl_fp16 = compute_ppl(model, test_ids, dev)
    hs_fp16 = hellaswag(model, tok, hex, dev)
    print(f"  fp16 baseline: PPL {ppl_fp16:.2f}, HellaSwag {hs_fp16*100:.1f}%", flush=True)

    budgets = {}
    for li in range(nL):
        budgets[li] = layer_budget(model.model.layers[li].mlp, calib[li], dev, Ls, Lpp, Lppp)
        print(f"  layer {li} budget done", flush=True)
    all_bits = [b for bk in budgets.values() for b in bk.values()]
    avg_bits = sum(all_bits)/len(all_bits)
    print(f"\n  certified budget: avg {avg_bits:.2f} bits ({avg_bits/16*100:.0f}% of fp16)", flush=True)

    quantize_model(model, budgets)
    ppl_cert = compute_ppl(model, test_ids, dev)
    hs_cert = hellaswag(model, tok, hex, dev)
    del model; gc.collect(); torch.cuda.empty_cache()

    # uniform baselines: 8-bit (memory-MATCHED to the certified budget, ~7 bits)
    # and 4-bit (more aggressive). 8-bit is the fair head-to-head: same ballpark
    # memory, so any quality gap isolates the value of the certified allocation.
    def uniform_eval(b):
        m = load_model(dev)
        quantize_model(m, {li: {e: b for e in range(m.model.layers[li].mlp.num_experts)}
                           for li in range(nL)})
        ppl = compute_ppl(m, test_ids, dev)
        hsa = hellaswag(m, tok, hex, dev)
        del m; gc.collect(); torch.cuda.empty_cache()
        return ppl, hsa
    ppl_u8, hs_u8 = uniform_eval(8)
    ppl_u4, hs_u4 = uniform_eval(4)

    print(f"\n{'='*64}\n  END-TO-END certified mixed-precision OLMoE (all 16 layers)\n{'='*64}")
    print(f"  {'config':28s} {'avg bits':>9s} {'mem':>6s} {'PPL':>8s} {'HellaSwag':>10s}")
    print(f"  {'fp16 (original)':28s} {16.0:>9.1f} {'100%':>6s} {ppl_fp16:>8.2f} {hs_fp16*100:>9.1f}%")
    print(f"  {'uniform 8-bit':28s} {8.0:>9.1f} {'50%':>6s} {ppl_u8:>8.2f} {hs_u8*100:>9.1f}%")
    print(f"  {'certified router-aware':28s} {avg_bits:>9.2f} {avg_bits/16*100:>5.0f}% {ppl_cert:>8.2f} {hs_cert*100:>9.1f}%")
    print(f"  {'uniform 4-bit':28s} {4.0:>9.1f} {'25%':>6s} {ppl_u4:>8.2f} {hs_u4*100:>9.1f}%")
    print(f"\n  delta_lyr={DELTA_LYR}, eps_L2={EPS}. Each layer carries a per-layer")
    print("  L2 certificate; end-to-end quality is measured (composition open).")


if __name__ == "__main__":
    main()
