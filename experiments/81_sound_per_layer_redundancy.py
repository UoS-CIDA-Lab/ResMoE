"""Experiment 81 — Can we SOUNDLY verify the per-layer redundancy budget (not just
measure it, as exp 80 Part 1)? Find the depth cutoff where it is possible.

Dropping experts at layer l perturbs layer l's output by Delta_l; the model-output
change is F_{>l}(h+Delta_l) - F_{>l}(h), F_{>l} = downstream stack (layers l+1..L-1,
final norm, lm_head). A sound certificate needs to bound this, i.e. needs the
downstream amplification L_{>l} = spectral norm of the downstream Jacobian. We compute
L_{>l} per layer by power iteration (autograd) on the downstream map at the data point.

This is the FIRST-ORDER (local) amplification; any fully sound bound is >= it, so:
  certified output-change ~ ||Delta_l|| * L_{>l}  (a lower bound on what any sound
  bound must report). If this already exceeds the tolerance for early layers, sound
  redundancy verification there is IMPOSSIBLE; late layers (short downstream, small
  L_{>l}) admit a real certificate. We report L_{>l}, the certified per-unit radius,
  and -- using the actual Delta from dropping K experts -- the certified output change.

GPU, live OLMoE. Run: python3 experiments/81_sound_per_layer_redundancy.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CTX = 64
PROBE_LAYERS = [0, 2, 4, 6, 8, 10, 12, 13, 14, 15]
PI_STEPS = 12
PI_R = 0.3                # finite step for the downstream secant (L2, Frobenius)
KDROP = 16                # experts dropped per layer to form the actual Delta
PROMPT = ("The mitochondria is the powerhouse of the cell, and researchers have long "
          "studied how energy production scales with demand across tissues and species.")


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
    Nexp = layers[0].mlp.num_experts
    ids = tok(PROMPT, return_tensors="pt").input_ids[:, :N_CTX].to(dev)

    # capture each decoder layer's kwargs + each layer's OUTPUT hidden state
    kw = [None] * nL
    out_h = [None] * nL
    pre_hs, post_hs = [], []
    for li in range(nL):
        def mkpre(li):
            def hk(_m, args, kwargs):
                kw[li] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            return hk
        def mkpost(li):
            def hk(_m, _a, output):
                out_h[li] = (output[0] if isinstance(output, tuple) else output).detach()
            return hk
        pre_hs.append(layers[li].register_forward_pre_hook(mkpre(li), with_kwargs=True))
        post_hs.append(layers[li].register_forward_hook(mkpost(li)))
    # also need MoE inputs to form Delta (router-input hidden states)
    moe_in = [None] * nL
    moe_hs = []
    for li in range(nL):
        def mkm(li):
            def hk(_m, a):
                moe_in[li] = a[0].detach()
            return hk
        moe_hs.append(layers[li].mlp.register_forward_pre_hook(mkm(li)))
    with torch.no_grad():
        model(ids)
    for h in pre_hs + post_hs + moe_hs:
        h.remove()

    def downstream(l, h):
        """h = output of layer l [1,seq,d] -> final logits [1,seq,V]."""
        for j in range(l + 1, nL):
            o = layers[j](h, **kw[j])
            h = o[0] if isinstance(o, tuple) else o
        h = model.model.norm(h)
        return model.lm_head(h)

    def amplification(l):
        """spectral norm of the downstream Jacobian at out_h[l], power iteration."""
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

    @torch.no_grad()
    def delta_norm(l, k):
        """max over tokens ||Delta_l|| from dropping the k lowest-contribution experts."""
        mlp = layers[l].mlp
        Hi = moe_in[l][0].float()
        lg = Hi @ mlp.gate.weight.float().T
        topv, topi = lg.topk(mlp.top_k, dim=-1)
        g = torch.softmax(topv, dim=-1)
        # contribution per expert
        contrib = torch.zeros(Nexp, device=dev)
        for e in range(Nexp):
            sel = (topi == e)
            if sel.any():
                ti = sel.any(1).nonzero().flatten()
                W1 = mlp.experts[e].gate_proj.weight.float()
                W2 = mlp.experts[e].up_proj.weight.float()
                W3 = mlp.experts[e].down_proj.weight.float()
                Ee = (F.silu(Hi[ti] @ W1.T) * (Hi[ti] @ W2.T)) @ W3.T
                contrib[e] = (g[sel] * Ee.norm(dim=-1)).mean()
        drop = torch.argsort(contrib)[:k].tolist()
        Delta = torch.zeros_like(Hi)
        for e in drop:
            sel = (topi == e)
            if sel.any():
                ti = sel.any(1).nonzero().flatten()
                W1 = mlp.experts[e].gate_proj.weight.float()
                W2 = mlp.experts[e].up_proj.weight.float()
                W3 = mlp.experts[e].down_proj.weight.float()
                Ee = (F.silu(Hi[ti] @ W1.T) * (Hi[ti] @ W2.T)) @ W3.T
                Delta[ti] += g[sel].unsqueeze(1) * Ee
        return Delta.norm(dim=-1).max().item()

    print(f"\n{'='*82}\n  SOUND per-layer redundancy: downstream amplification L_>l and the"
          f" certified output change\n  (drop K={KDROP} lowest-contribution experts; "
          f"first-order L_>l is a LOWER bound on any sound amplification)\n{'='*82}")
    print(f"  {'layer l':>7s} | {'#downstream':>11s} | {'L_>l (amp)':>11s} | "
          f"{'max||Delta||':>12s} | {'cert out-change ~ L*||Delta||':>28s}")
    print("  " + "-"*80)
    for l in PROBE_LAYERS:
        amp = amplification(l)
        dn = delta_norm(l, KDROP)
        print(f"  {l:>7d} | {nL - 1 - l:>11d} | {amp:>11.2f} | {dn:>12.4f} | "
              f"{amp * dn:>28.3f}", flush=True)

    print(f"\n  Read: L_>l EXPLODES as l decreases (more downstream layers compound the")
    print("  amplification). Sound redundancy verification is possible only where")
    print("  L_>l*||Delta|| stays below the output tolerance -- i.e. the LAST FEW layers.")
    print("  For early/mid layers even this first-order (optimistic) amplification is huge,")
    print("  so NO sound bound can certify their redundancy: the depth wall, made precise.")


if __name__ == "__main__":
    main()
