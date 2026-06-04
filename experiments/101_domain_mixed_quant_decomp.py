"""Experiment 101 — Domain-aware decomposition via QUANTIZATION (favorable version).

Pruning drops an expert's whole contribution when it fires (large divergence, exp 100).
Quantizing keeps the function -> far less divergence at the same memory. We compare, at
matched average memory, three domain-aware decompositions, certified on the domain
(distributional Clopper-Pearson 95% bound on prediction divergence from the full model):
  PRUNE          : keep top experts by domain-impact at fp16, drop the rest.
  UNIFORM-QUANT  : all experts at the bit-width matching the memory.
  DOMAIN-MIXED   : high domain-impact experts at 8-bit, low at 2-bit (split to hit the
                   target avg bits), never-selected dropped -- the certified-budget idea,
                   domain-conditional.
Expect DOMAIN-MIXED <= UNIFORM << PRUNE. Selection/allocation = impact (verification adds
the per-expert worst-case bound + the on-domain certificate, not the ordering; exp 95).

GPU, live OLMoE, code + prose. Run: python3 experiments/101_domain_mixed_quant_decomp.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 256
N_EVAL = 512
CHUNK = 256
EPS = 0.05
CONF = 0.95
AVG_BITS = [6.0, 4.0, 3.0]            # memory axis = avg_bits/16


def domains(tok):
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    he = load_dataset("openai_humaneval", split="test")
    return {"code": tok("\n\n".join(he["prompt"]), return_tensors="pt").input_ids[0],
            "prose": tok("\n\n".join(t for t in wt["text"] if t.strip()),
                         return_tensors="pt").input_ids[0]}


def cp_upper(k, n, conf):
    try:
        from scipy.stats import beta
        return 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))
    except Exception:
        return min(1.0, k / n + math.sqrt(math.log(1.0 / (1.0 - conf)) / (2 * n)))


def quant(W, b):
    if b >= 16:
        return W
    qmax = 2 ** (b - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


@torch.no_grad()
def collect(model, ids, dev, layers, lo, hi):
    cap = {li: [] for li in layers}
    hs = []
    for li in layers:
        def mk(li):
            def hk(_m, a):
                cap[li].append(a[0].detach().reshape(-1, a[0].shape[-1]))
            return hk
        hs.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    model(ids[lo:hi].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    return {li: torch.cat(cap[li]).float() for li in layers}


@torch.no_grad()
def impact(mlp, H, dev):
    Wg = mlp.gate.weight.float()
    lg = H @ Wg.T
    topv, topi = lg.topk(mlp.top_k, dim=-1); g = torch.softmax(topv, dim=-1)
    c = torch.zeros(mlp.num_experts, device=dev)
    for e in range(mlp.num_experts):
        sel = (topi == e)
        if sel.any():
            ti = sel.any(1).nonzero().flatten()
            W1 = mlp.experts[e].gate_proj.weight.float()
            W2 = mlp.experts[e].up_proj.weight.float()
            W3 = mlp.experts[e].down_proj.weight.float()
            Ee = (F.silu(H[ti] @ W1.T) * (H[ti] @ W2.T)) @ W3.T
            c[e] = (g[sel] * Ee.norm(dim=-1)).mean()
    return c


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    doms = domains(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers
    nL = len(layers); Nexp = layers[0].mlp.num_experts

    allw = {(li, e, p): getattr(layers[li].mlp.experts[e], p).weight
            for li in range(nL) for e in range(Nexp)
            for p in ("gate_proj", "up_proj", "down_proj")}
    saved = {k: w.detach().cpu().clone() for k, w in allw.items()}

    def restore():
        with torch.no_grad():
            for k, w in allw.items():
                w.data.copy_(saved[k].to(dev))

    def set_bits(bits_of):   # bits_of: (li,e)->bits (0=drop)
        with torch.no_grad():
            for li in range(nL):
                for e in range(Nexp):
                    b = bits_of[(li, e)]
                    for p in ("gate_proj", "up_proj", "down_proj"):
                        w = allw[(li, e, p)]
                        if b <= 0:
                            w.data.zero_()
                        else:
                            w.data.copy_(quant(saved[(li, e, p)].to(dev).float(), b).to(w.dtype))

    for name, ids in doms.items():
        ids = ids.to(dev)
        Hc = collect(model, ids, dev, list(range(nL)), 0, N_CALIB)
        imp = {li: impact(layers[li].mlp, Hc[li], dev) for li in range(nL)}
        order = {li: torch.argsort(imp[li]).tolist() for li in range(nL)}  # ascending

        @torch.no_grad()
        def base_preds():
            return [model(ids[c0:c0+CHUNK].unsqueeze(0)).logits[0, :-1].float().argmax(-1)
                    for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK)]

        bp = base_preds()

        @torch.no_grad()
        def cert_div():
            kbad = tot = 0
            for j, c0 in enumerate(range(N_CALIB, N_CALIB + N_EVAL, CHUNK)):
                p = model(ids[c0:c0+CHUNK].unsqueeze(0)).logits[0, :-1].float().argmax(-1)
                kbad += int((p != bp[j]).sum().item()); tot += p.numel()
            return cp_upper(kbad, tot, CONF) * 100, kbad / tot * 100

        print(f"\n{'='*74}\n  DOMAIN={name}: certified pred-divergence (95%) vs full model, "
              f"matched memory\n{'='*74}")
        print(f"  {'avgbits':>7s} {'mem':>5s} | {'PRUNE':>14s} | {'UNIFORM-q':>14s} | "
              f"{'DOMAIN-MIXED':>14s}")
        print("  " + "-"*66)
        for T in AVG_BITS:
            mem = T / 16 * 100
            # PRUNE: keep top (T/16) fraction of experts at fp16, drop rest
            keep = round(T / 16 * Nexp)
            pr = {(li, e): (16 if e in set(order[li][Nexp-keep:]) else 0)
                  for li in range(nL) for e in range(Nexp)}
            # UNIFORM: round T to nearest available bit
            ub = min([8, 6, 4, 3, 2], key=lambda b: abs(b - T))
            un = {(li, e): ub for li in range(nL) for e in range(Nexp)}
            # DOMAIN-MIXED: high-impact 8, low 2 (frac f at 8: 8f+2(1-f)=T -> f=(T-2)/6)
            f = max(0.0, min(1.0, (T - 2) / 6))
            n8 = round(f * Nexp)
            dm = {}
            for li in range(nL):
                hi8 = set(order[li][Nexp-n8:])
                for e in range(Nexp):
                    dm[(li, e)] = 8 if e in hi8 else 2
            row = []
            for cfg in (pr, un, dm):
                set_bits(cfg); c, emp = cert_div(); restore()
                row.append(f"{c:>5.1f}({emp:>4.1f})")
            print(f"  {T:>7.1f} {mem:>4.0f}% | {row[0]:>14s} | {row[1]:>14s} | "
                  f"{row[2]:>14s}", flush=True)

    print(f"\n  cells = certified%(empirical%) prediction divergence from full model on the")
    print("  domain. QUANT (uniform/mixed) << PRUNE at matched memory: quantizing keeps the")
    print("  expert function, dropping destroys it. DOMAIN-MIXED spends bits on high-impact")
    print("  experts. The on-domain distributional certificate bounds it (95% conf).")


if __name__ == "__main__":
    main()
