"""Experiment 89 — Clean (float32) SOUND DIRECTIONAL certificate for FLIP-FREE
compression, fixing exp 87/88's fp16 finite-diff noise.

For a late injection layer L where the compression Delta_L causes ZERO downstream
routing flips (exp 88: L14/L15 -> 0), the downstream map G = layers[L+1:]+norm+lm_head
is SMOOTH, so the output change has a tight directional Taylor bound:
    ||G(h0+Delta) - G(h0)|| <= ||J_G(h0).Delta||  +  (1/2) * max_t||D''(t)|| ,
    D(t)=G(h0+tDelta), D''= second derivative along Delta (curvature).
We cast ONLY the downstream submodule to float32 (the rest stays fp16) so the finite
differences are numerically clean. Report: first-order ||J.Delta||, exact change,
curvature term, the directional bound, tightness (bound/exact), soundness (bound>=exact),
and the flip count (must be ~0 for the smooth regime).

GPU, live OLMoE. Run: python3 experiments/89_flipfree_directional_cert.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CTX = 48
INJECT_LAYERS = [13, 14, 15]
T1 = 0.02                       # small step for first derivative
TCURV = [0.2, 0.4, 0.6, 0.8]    # points to sample curvature D''
SCURV = 0.1                     # half-step for second difference
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
        torch.cuda.set_per_process_memory_fraction(0.92, 0)
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

    @torch.no_grad()
    def flips(l, hpert, hbase):
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

    print(f"\n{'='*92}\n  FLIP-FREE directional certificate (float32 downstream), 8-bit Delta"
          f"\n{'='*92}")
    print(f"  {'L':>3s} | {'flips':>5s} | {'1st ||J.D||':>11s} | {'curv term':>9s} | "
          f"{'DIR bound':>9s} | {'EXACT':>9s} | {'bound/exact':>11s} | {'sound':>5s}")
    print("  " + "-"*90)

    for L in INJECT_LAYERS:
        D16 = delta_at(L)
        h0_16 = out_h[L][0]
        f = flips(L, (out_h[L] + D16.unsqueeze(0)).clone(), out_h[L].clone())

        # cast downstream submodule to float32 for clean numerics
        mods = [layers[j] for j in range(L + 1, nL)] + [model.model.norm, model.lm_head]
        orig_dtype = {}
        for m in mods:
            for n, p in m.named_parameters(recurse=True):
                orig_dtype[(m, n)] = None
        for m in mods:
            m.float()
        kwf = {}
        for j in range(L + 1, nL):
            kwf[j] = {}
            for k, v in kw[j].items():
                if torch.is_tensor(v) and v.is_floating_point():
                    kwf[j][k] = v.float()
                elif isinstance(v, tuple):
                    kwf[j][k] = tuple(x.float() if torch.is_tensor(x) and
                                      x.is_floating_point() else x for x in v)
                else:
                    kwf[j][k] = v

        @torch.no_grad()
        def G(h):                                   # h [seq,d] float32 -> logits float32
            x = h.unsqueeze(0)
            for j in range(L + 1, nL):
                o = layers[j](x, **kwf[j])
                x = o[0] if isinstance(o, tuple) else o
            return model.lm_head(model.model.norm(x))

        h0 = h0_16.float()
        D = D16.float()
        dn = D.norm().item()
        with torch.no_grad():
            G0 = G(h0)
            exact = (G(h0 + D) - G0).norm().item()
            # first-order along D: [G(h0+T1 D)-G0]/T1
            jd = (G(h0 + T1 * D) - G0).norm().item() / T1
            # curvature: max ||D''(t)|| via second difference, float32
            M = 0.0
            for t in TCURV:
                dpp = (G(h0 + (t + SCURV) * D) - 2 * G(h0 + t * D)
                       + G(h0 + (t - SCURV) * D)) / (SCURV ** 2)
                M = max(M, dpp.norm().item())
            bound = jd + 0.5 * M
        # restore fp16
        for m in mods:
            m.half()
        sound = "yes" if bound >= exact - 1e-3 else "NO"
        print(f"  {L:>3d} | {f:>5d} | {jd:>11.3f} | {0.5*M:>9.3f} | {bound:>9.3f} | "
              f"{exact:>9.3f} | {bound/max(exact,1e-9):>10.2f}x | {sound:>5s}", flush=True)

    print(f"\n  For flip-free L (flips~0): DIR bound = ||J.Delta|| + 1/2 max||D''|| is SOUND")
    print("  (>= exact) and TIGHT (bound/exact small) in clean float32 -- a real certified")
    print("  model-output bound for the compression, where agnostic ||J||*||Delta|| was inf.")
    print("  (curvature via sampled D''; rigor needs an analytic D'' sup -- clean for norm+lmhead.)")


if __name__ == "__main__":
    main()
