"""Experiment 75 — Feasibility of CERTIFIED data-region compression: does a sound
bound survive depth, or does per-layer amplification blow it up?

The user's pivot: MoE is robust because of redundancy -> redundancy = removable
slack -> can verification certify that stripping it (prune/quant) preserves behavior
ON THE DATA MANIFOLD (not an adversarial ball, which we showed is vacuous)? The make-
or-break quantity is the per-layer WORST-CASE amplification L_l (local Lipschitz of
the layer map over the data): a sound bound grows as prod_l L_l, so
  L_l ~ 1.0-1.1  -> prod over 16 layers ~ few x   -> certified data-region compression VIABLE
  L_l >~ 1.4     -> prod ~ hundreds x             -> wall is fundamental, close the idea.
Residual connections make a layer h -> h + sublayer(h), so L_l could be near 1 + small.

We measure, per layer l (original model, over calib data):
  (A) EMPIRICAL DRIFT D_l = ||H_compressed[l] - H_original[l]||  (actual effect of
      compression entering layer l), for quant-8bit AND prune-1-expert.
  (B) ACTUAL amplification a_l = D_{l+1}/D_l (how the real drift grows).
  (C) WORST-CASE L_l = max_{||delta||=r} ||layer_l(h+delta) - layer_l(h)|| / r  (PGD
      power-iteration on the single layer at the data point) -- what a SOUND bound
      must assume. Gap a_l << L_l => the real drift is benign but a sound bound is
      forced to assume worst-case => loose; L_l itself >1.4 => blows up regardless.

GPU, live OLMoE. Run: python3 experiments/75_depth_amplification.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 48
LIP_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14]
LIP_STEPS = 25
LIP_R = 0.5          # finite L2 step (Frobenius over the sequence) for the secant slope
PROMPT = ("The mitochondria is the powerhouse of the cell, and researchers have long "
          "studied how energy production scales with demand across tissues and species, "
          "from the smallest insects to the largest whales, under varying conditions.")


def quant(W, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


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
    ids = tok(PROMPT, return_tensors="pt").input_ids[:, :N_CALIB].to(dev)

    # ---- capture per-layer input hidden states + the kwargs each layer receives ----
    cap_h, cap_kw = [None]*nL, [None]*nL

    def mk(li):
        def hk(_m, args, kwargs):
            cap_h[li] = args[0].detach()
            cap_kw[li] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        return hk
    handles = [layers[li].register_forward_pre_hook(mk(li), with_kwargs=True)
               for li in range(nL)]
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    H_orig = [cap_h[li] for li in range(nL)]   # each [1, seq, d]

    # also need the final hidden (input to norm/lm_head) = output of last layer
    @torch.no_grad()
    def all_layer_inputs():
        cap = [None]*nL
        hs = []
        for li in range(nL):
            def mk2(li):
                def hk(_m, a):
                    cap[li] = a[0].detach()
                return hk
            hs.append(layers[li].register_forward_pre_hook(mk2(li)))
        model(ids)
        for h in hs:
            h.remove()
        return cap

    # ---- (C) per-layer worst-case local Lipschitz on the ORIGINAL model ----
    def layer_fn(li, h):
        out = layers[li](h, **cap_kw[li])
        return out[0] if isinstance(out, tuple) else out

    def lipschitz(li):
        h0 = H_orig[li]
        with torch.no_grad():
            base = layer_fn(li, h0)
        v = torch.randn_like(h0)
        v = v / v.norm()
        best = 0.0
        for _ in range(LIP_STEPS):
            v = v.detach().requires_grad_(True)
            pert = layer_fn(li, h0 + LIP_R * v)
            obj = ((pert - base) ** 2).sum()
            best = max(best, obj.detach().item() ** 0.5 / LIP_R)
            g, = torch.autograd.grad(obj, v)
            with torch.no_grad():
                v = g / (g.norm() + 1e-12)
        return best

    print(f"  computing per-layer worst-case Lipschitz (r={LIP_R}) ...", flush=True)
    Lip = {}
    for li in LIP_LAYERS:
        Lip[li] = lipschitz(li)
        print(f"    layer {li:2d}: L_l = {Lip[li]:.3f}", flush=True)

    # ---- (A) empirical drift for each compression ----
    # save expert weights (CPU) to restore after each compression
    expert_w = []
    for layer in layers:
        for e in range(layer.mlp.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                expert_w.append(getattr(layer.mlp.experts[e], proj).weight)
    saved = [w.detach().cpu().clone() for w in expert_w]

    def restore():
        with torch.no_grad():
            for w, s in zip(expert_w, saved):
                w.data.copy_(s.to(dev))

    def apply_quant8():
        with torch.no_grad():
            for w in expert_w:
                w.data.copy_(quant(w.float(), 8).to(torch.float16))

    def apply_prune1():
        # per layer, zero the down_proj of the least-frequently-routed expert
        with torch.no_grad():
            for li in range(nL):
                mlp = layers[li].mlp
                lg = (H_orig[li][0].float() @ mlp.gate.weight.float().T)
                topk = lg.topk(mlp.top_k, dim=-1).indices
                cnt = torch.bincount(topk.flatten(), minlength=mlp.num_experts)
                e = int(cnt.argmin().item())
                mlp.experts[e].down_proj.weight.data.zero_()

    def drift_traj():
        comp_in = all_layer_inputs()
        D = []
        for li in range(nL):
            d = (comp_in[li] - H_orig[li])[0]           # [seq, d]
            D.append(d.norm(dim=-1).mean().item())       # mean per-token L2 drift
        return D

    results = {}
    for name, apply in [("quant8", apply_quant8), ("prune1", apply_prune1)]:
        restore()
        apply()
        results[name] = drift_traj()
        restore()
        print(f"  drift trajectory ({name}) done", flush=True)

    # ---- report ----
    print(f"\n{'='*80}\n  DEPTH AMPLIFICATION (mean per-token L2 drift; "
          f"L_l = worst-case layer Lipschitz)\n{'='*80}")
    print(f"  {'layer':>5s} | {'L_l(worst)':>10s} | "
          f"{'quant8 drift':>12s} {'a_l':>6s} | {'prune1 drift':>12s} {'a_l':>6s}")
    print("  " + "-"*72)
    for li in range(nL):
        lstr = f"{Lip[li]:.3f}" if li in Lip else "  -  "
        q = results["quant8"]; p = results["prune1"]
        aq = q[li]/q[li-1] if li > 0 and q[li-1] > 1e-9 else float("nan")
        ap = p[li]/p[li-1] if li > 0 and p[li-1] > 1e-9 else float("nan")
        print(f"  {li:>5d} | {lstr:>10s} | {q[li]:>12.4f} {aq:>6.2f} | "
              f"{p[li]:>12.4f} {ap:>6.2f}", flush=True)

    Lprod = 1.0
    for li in LIP_LAYERS:
        Lprod *= Lip[li]
    geom = (Lprod) ** (1.0 / len(LIP_LAYERS))
    print(f"\n  geom-mean worst-case L_l = {geom:.3f}  ->  over {nL} layers ~ {geom**nL:.1f}x")
    print(f"  actual drift growth: quant8 {results['quant8'][-1]/max(results['quant8'][1],1e-9):.1f}x, "
          f"prune1 {results['prune1'][-1]/max(results['prune1'][1],1e-9):.1f}x (layers 1->{nL-1})")
    print("\n  VIABLE if geom-mean L_l ~ 1.0-1.1 (sound bound ~ few x over depth).")
    print("  WALL if L_l >~ 1.4 (sound bound explodes); and a_l << L_l shows how much")
    print("  tighter the real drift is than any worst-case-direction sound bound.")


if __name__ == "__main__":
    main()
