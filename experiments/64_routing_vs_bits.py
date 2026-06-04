"""Experiment 64 — Certified routing equivalence vs quantization aggressiveness.

exp 63 showed the certified router-aware budget (delta=0.01, mixes in 2-bit experts)
induces hidden-state drift (~2.3 L2 deep) that swamps the router margin -> only 6.4%
of routing decisions are CERTIFIABLY preserved (89% empirically). HYPOTHESIS: drift
scales with quantization aggressiveness, so NEAR-LOSSLESS uniform quantization (8-bit,
per-expert dev ~100x smaller) yields ~100x smaller drift, comparable to the deeper
layers' router margins (~0.14) -> a SUBSTANTIAL certified-preservation fraction. This
maps out the compression-vs-certifiable-structural-fidelity tradeoff: uniform {8,6,4}.

Routing-orthogonal: the router (linear) is never quantized, so its margin radius is
fixed; only the drift moves with bits. Certified-preserved at (token,layer) iff the
router margin radius (from the linear-router logit intervals) exceeds the measured
drift d_l = ||h_l^q - h_l||_2 -> top-K SET provably invariant -> same experts routed.

One model in memory at a time (reload per bit-width). GPU, 75% cap.
Run: python3 experiments/64_routing_vs_bits.py
"""
from __future__ import annotations

import sys
import pathlib
import gc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
BIT_SWEEP = [8, 6, 4, 3]
N_EVAL = 256


def quant(W, bits):
    if bits >= 16:
        return W
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def load_model(dev):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    return m.to(dev).eval()


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
def quantize_uniform(model, b):
    for layer in model.model.layers:
        for ex in layer.mlp.experts:
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
    rn = rownorm
    radii = torch.empty(T, device=logits.device)
    for t in range(T):
        S = inset[t]
        ls = logits[t][S]; lu = logits[t][~S]
        ws = rn[S]; wu = rn[~S]
        num = ls[:, None] - lu[None, :]
        den = ws[:, None] + wu[None, :]
        radii[t] = (num / den).min()
    return radii


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_ids = tok("\n\n".join(t for t in test["text"] if t.strip()),
                   return_tensors="pt").input_ids[0]

    print("Loading fp16 reference routing...", flush=True)
    model = load_model(dev)
    nL = len(model.model.layers)
    K = model.model.layers[0].mlp.top_k
    N = model.model.layers[0].mlp.num_experts
    H_fp = collect_hidden(model, eval_ids, dev, nL, N_EVAL)
    route_fp = {li: topk_sets(model.model.layers[li].mlp, H_fp[li]) for li in range(nL)}
    rownorm = {li: model.model.layers[li].mlp.gate.weight.float().norm(dim=1)
               for li in range(nL)}
    # precompute the certified routing radius per (layer,token) at the fp16 state
    radii = {li: cert_radius(route_fp[li][1], route_fp[li][0], rownorm[li], K, N)
             for li in range(nL)}
    del model; gc.collect(); torch.cuda.empty_cache()

    results = []
    for b in BIT_SWEEP:
        model = load_model(dev)
        quantize_uniform(model, b)
        H_q = collect_hidden(model, eval_ids, dev, nL, N_EVAL)
        route_q = {li: topk_sets(model.model.layers[li].mlp, H_q[li]) for li in range(nL)}
        del model; gc.collect(); torch.cuda.empty_cache()

        tot_emp = tot_cert = tot_n = 0
        drift_by_layer = []
        for li in range(nL):
            topk_fp = route_fp[li][0]; topk_q = route_q[li][0]
            T = min(topk_fp.shape[0], topk_q.shape[0])
            sfp = [set(topk_fp[t].tolist()) for t in range(T)]
            sq = [set(topk_q[t].tolist()) for t in range(T)]
            emp = sum(sfp[t] == sq[t] for t in range(T))
            d = (H_q[li][:T] - H_fp[li][:T]).norm(dim=1)
            cert = (radii[li][:T] > d).sum().item()
            tot_emp += emp; tot_cert += cert; tot_n += T
            drift_by_layer.append(d.mean().item())
        results.append((b, tot_emp/tot_n, tot_cert/tot_n,
                        sum(drift_by_layer)/len(drift_by_layer),
                        max(drift_by_layer)))
        print(f"  {b}-bit: emp agree {tot_emp/tot_n*100:.1f}%, "
              f"certified-preserved {tot_cert/tot_n*100:.1f}%, "
              f"drift mean {sum(drift_by_layer)/len(drift_by_layer):.3f} "
              f"max {max(drift_by_layer):.3f}", flush=True)

    print(f"\n{'='*72}\n  Certified routing equivalence vs quantization bits (uniform)"
          f"\n{'='*72}")
    print(f"  {'bits':>5s} {'mem':>5s} {'emp top-K agree':>16s} "
          f"{'CERTIFIED-preserved':>20s} {'drift mean/max':>16s}")
    for b, emp, cert, dm, dM in results:
        print(f"  {b:>5d} {b/16*100:>4.0f}% {emp*100:>15.1f}% {cert*100:>19.1f}% "
              f"{dm:>7.3f}/{dM:<7.3f}")
    print(f"\n  router-aware delta=0.01 (exp 63, 50% mem): emp 89.2%, certified 6.4%.")
    print("  CERTIFIED-preserved = (token,layer) where router top-K SET is PROVABLY")
    print("  invariant under the measured quant drift (router linear, tight). The")
    print("  compression<->certifiable-structural-fidelity tradeoff for MoE quant.")


if __name__ == "__main__":
    main()
