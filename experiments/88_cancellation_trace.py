"""Experiment 88 — Quantify CANCELLATION and FLIPS along the compression direction
(clean replacement for exp 87's fp16-finite-diff noise).

For injecting compression Delta_l at layer l, trace the downstream output change along
the path h0 + t*Delta for t in [0..1]:
    c(t) = ||G(h0 + t*Delta) - G(h0)||   (Frobenius over logits; O(t), fp16-robust)
and the cumulative routing FLIPS up to t. Reading:
  - c(t) LINEAR in t (c(t)/t ~ const)         -> first-order directional is tight (smooth).
  - c(t) SUBLINEAR (c(t)/t decreasing)         -> CANCELLATION (small net change from
        large opposing first/higher-order terms) -> a tight SOUND bound must capture it.
  - jumps in c(t) / flips>0                     -> routing discontinuity along the path.
We report c(t) and flips(t), and the ratio [c(0.1)/0.1] (early slope = directional
derivative estimate) vs c(1.0) (actual) -> how much cancellation.

GPU, live OLMoE. Run: python3 experiments/88_cancellation_trace.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CTX = 48
INJECT_LAYERS = [4, 8, 12, 14, 15]
TGRID = [0.1, 0.25, 0.5, 0.75, 1.0]
PROMPT = ("The mitochondria is the powerhouse of the cell, and researchers have long "
          "studied how energy production scales with demand across many tissues.")


def quant8(W):
    qmax = 127
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-128, qmax) * s


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    nL = len(layers)
    ids = tok(PROMPT, return_tensors="pt").input_ids[:, :N_CTX].to(dev)

    kw = [None] * nL
    out_h = [None] * nL
    ph, qh = [], []
    for li in range(nL):
        def mkpre(li):
            def hk(_m, a, k):
                kw[li] = {kk: v for kk, v in k.items() if kk != "hidden_states"}
            return hk
        def mkpost(li):
            def hk(_m, _a, o):
                out_h[li] = (o[0] if isinstance(o, tuple) else o).detach()
            return hk
        ph.append(layers[li].register_forward_pre_hook(mkpre(li), with_kwargs=True))
        qh.append(layers[li].register_forward_hook(mkpost(li)))
    with torch.no_grad():
        model(ids)
    for h in ph + qh:
        h.remove()

    @torch.no_grad()
    def downstream(l, h):
        for j in range(l + 1, nL):
            o = layers[j](h, **kw[j])
            h = o[0] if isinstance(o, tuple) else o
        return model.lm_head(model.model.norm(h))

    @torch.no_grad()
    def flips_upto(l, hpert, hbase):
        f = 0
        ha, hb = hpert, hbase
        for j in range(l + 1, nL):
            oa = layers[j](ha, **kw[j]); ha = oa[0] if isinstance(oa, tuple) else oa
            ob = layers[j](hb, **kw[j]); hb = ob[0] if isinstance(ob, tuple) else ob
            if j + 1 < nL:
                g = layers[j + 1].mlp.gate.weight.float()
                k = layers[j + 1].mlp.top_k
                ta = (ha[0].float() @ g.T).topk(k, -1).indices
                tb = (hb[0].float() @ g.T).topk(k, -1).indices
                f += int((ta != tb).any(-1).sum().item())
        return f

    def delta_at(l):
        mlp = layers[l].mlp
        saved = {}
        for e in range(mlp.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                w = getattr(mlp.experts[e], proj).weight
                saved[(e, proj)] = w.data.clone()
                w.data.copy_(quant8(w.float()).to(w.dtype))
        cap = {}
        def hk(_m, _a, o):
            cap["o"] = (o[0] if isinstance(o, tuple) else o).detach()
        hnd = layers[l].register_forward_hook(hk)
        with torch.no_grad():
            model(ids)
        hnd.remove()
        for (e, proj), w in saved.items():
            getattr(mlp.experts[e], proj).weight.data.copy_(w)
        return (cap["o"] - out_h[l])[0]

    print(f"\n{'='*88}\n  CANCELLATION / FLIP trace along h0 + t*Delta (8-bit Delta); "
          f"c(t)=||G(h0+tD)-G(h0)||\n{'='*88}")
    for l in INJECT_LAYERS:
        D = delta_at(l).unsqueeze(0)
        h0 = out_h[l]
        cs, fs = [], []
        with torch.no_grad():
            G0 = downstream(l, h0)
            for t in TGRID:
                Gt = downstream(l, h0 + t * D)
                cs.append((Gt - G0).norm().item())
                fs.append(flips_upto(l, (h0 + t * D).clone(), h0.clone()))
        early_slope = cs[0] / TGRID[0]            # ~ directional derivative magnitude
        ratio = early_slope / max(cs[-1], 1e-9)   # extrapolated/actual: >1 = cancellation
        print(f"\n  layer {l}:  ||Delta||={D.norm().item():.3f}")
        print("    t      : " + "  ".join(f"{t:>6.2f}" for t in TGRID))
        print("    c(t)   : " + "  ".join(f"{c:>6.2f}" for c in cs))
        print("    flips  : " + "  ".join(f"{f:>6d}" for f in fs))
        shape = ("LINEAR (1st-order tight)" if 0.8 <= ratio <= 1.25
                 else f"SUBLINEAR x{ratio:.1f} (cancellation)" if ratio > 1.25
                 else "superlinear")
        print(f"    early-slope c(0.1)/0.1 = {early_slope:.2f} vs actual c(1)={cs[-1]:.2f}"
              f"  -> {shape}", flush=True)

    print(f"\n  LINEAR + flips=0 => directional first-order IS the tight sound target.")
    print("  SUBLINEAR => the small net change is from cancellation (hard to bound sound).")
    print("  flips>0 along the path => routing discontinuity contributes to c(t).")


if __name__ == "__main__":
    main()
