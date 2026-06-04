"""Experiment 86 — DIRECTIONAL (difference-aware) vs AGNOSTIC (||J||*||Delta||) model-
level bound, and where routing flips break it.

The user's point: don't bound the worst-case MAGNITUDE of the perturbation (||J||*eps,
vacuous, exp 80/81); bound the KNOWN difference direction Delta. We inject a compression
difference Delta_l at layer l's output and propagate through the ORIGINAL downstream
stack, comparing:
  EMPIRICAL  = ||downstream(h_l + Delta_l) - downstream(h_l)||  (the TRUE directional
               output change -- what a tight difference-aware bound targets),
  AGNOSTIC   = ||J_downstream||_2 * ||Delta_l||  (direction-agnostic worst-case, exp 81),
  FLIPS      = # downstream top-K routing decisions that differ between the two
               trajectories (the discontinuity that breaks a smooth directional bound).
If EMPIRICAL << AGNOSTIC -> the vacuousness was the AGNOSTIC bound's fault (direction
matters), validating the user. FLIPS>0 -> the residual obstruction is the routing
discontinuity, not magnitude. We use 8-bit quant of layer l as the (small) Delta_l.

GPU, live OLMoE. Run: python3 experiments/86_directional_vs_agnostic.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CTX = 48
INJECT_LAYERS = [4, 8, 11, 13, 14, 15]
PI_STEPS = 12
PI_R = 0.3
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

    # capture layer kwargs + each layer's OUTPUT
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

    def downstream(l, h):
        for j in range(l + 1, nL):
            o = layers[j](h, **kw[j])
            h = o[0] if isinstance(o, tuple) else o
        return model.lm_head(model.model.norm(h))

    @torch.no_grad()
    def routing_flips(l, h_a, h_b):
        """count top-K routing decisions differing between two trajectories, layers>l."""
        flips = 0
        ha, hb = h_a, h_b
        for j in range(l + 1, nL):
            mlp = layers[j].mlp
            # router input is post-attention-norm inside the layer; approximate by
            # running the layer and reading its mlp router on each trajectory
            # simpler: compare top-K of the layer's mlp gate on the layer input
            for h_in, store in ((ha, "a"), (hb, "b")):
                pass
            oa = layers[j](ha, **kw[j]); ha = oa[0] if isinstance(oa, tuple) else oa
            ob = layers[j](hb, **kw[j]); hb = ob[0] if isinstance(ob, tuple) else ob
            # compare router decisions at layer j+1 input via the mlp gate
            if j + 1 < nL:
                g = layers[j + 1].mlp.gate.weight.float()
                ta = (ha[0].float() @ g.T).topk(mlp.top_k, -1).indices
                tb = (hb[0].float() @ g.T).topk(mlp.top_k, -1).indices
                flips += int((ta != tb).any(-1).sum().item())  # tokens with changed set
        return flips

    # 8-bit quant of each injection layer's experts (compute Delta_l locally)
    def delta_at(l):
        mlp = layers[l].mlp
        # original layer-l output already in out_h[l]; compute quantized layer-l output
        saved = {}
        for e in range(mlp.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                w = getattr(mlp.experts[e], proj).weight
                saved[(e, proj)] = w.data.clone()
                w.data.copy_(quant8(w.float()).to(w.dtype))
        with torch.no_grad():
            inp = out_h[l - 1] if l > 0 else None
            # re-run layer l on its original input to get quantized output
            # original input to layer l = out_h[l-1] (or embeddings for l=0)
        # restore later; we need layer-l input. Recompute by running up to l is costly;
        # instead approximate Delta as (quant expert effect). Use layer recompute:
        # run the full model with only layer l quantized, read out_h[l].
        cap = {}
        def hk(_m, _a, o):
            cap["o"] = (o[0] if isinstance(o, tuple) else o).detach()
        hnd = layers[l].register_forward_hook(hk)
        with torch.no_grad():
            model(ids)
        hnd.remove()
        q_out = cap["o"]
        for (e, proj), w in saved.items():
            getattr(layers[l].mlp.experts[e], proj).weight.data.copy_(w)
        return (q_out - out_h[l])[0]      # [seq, d] Delta at layer l output

    def amplification(l):
        h0 = out_h[l]
        with torch.no_grad():
            base = downstream(l, h0)
        v = torch.randn_like(h0); v = v / v.norm()
        best = 0.0
        for _ in range(PI_STEPS):
            v = v.detach().requires_grad_(True)
            pert = downstream(l, h0 + PI_R * v)
            obj = ((pert - base) ** 2).sum()
            best = max(best, obj.detach().item() ** 0.5 / PI_R)
            g, = torch.autograd.grad(obj, v)
            with torch.no_grad():
                v = g / (g.norm() + 1e-12)
        return best

    print(f"\n{'='*86}\n  DIRECTIONAL (true diff) vs AGNOSTIC (||J||*||Delta||) output bound;"
          f" 8-bit quant Delta\n{'='*86}")
    print(f"  {'inj layer':>9s} | {'||Delta_l||':>11s} | {'EMPIRICAL out-change':>20s} | "
          f"{'AGNOSTIC ||J||*||D||':>20s} | {'ratio':>7s} | {'route flips':>11s}")
    print("  " + "-"*84)
    for l in INJECT_LAYERS:
        D = delta_at(l)                              # [seq,d]
        dn = D.norm(dim=-1).max().item()
        h0 = out_h[l]
        with torch.no_grad():
            emp = (downstream(l, h0 + D.unsqueeze(0)) - downstream(l, h0)).abs().max().item()
        amp = amplification(l)
        agn = amp * dn
        with torch.no_grad():
            flips = routing_flips(l, (h0 + D.unsqueeze(0)).clone(), h0.clone())
        print(f"  {l:>9d} | {dn:>11.4f} | {emp:>20.4f} | {agn:>20.2f} | "
              f"{agn/max(emp,1e-9):>6.0f}x | {flips:>11d}", flush=True)

    print(f"\n  EMPIRICAL << AGNOSTIC => the vacuousness was the AGNOSTIC (magnitude) bound;")
    print("  the TRUE directional change is small (a difference-aware bound targets it).")
    print("  route flips > 0 => the residual obstruction to a SOUND directional bound is")
    print("  the routing discontinuity (the difference jumps at a flip), not magnitude.")


if __name__ == "__main__":
    main()
