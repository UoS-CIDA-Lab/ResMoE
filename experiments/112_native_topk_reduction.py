"""Experiment 112 — Sanity check on the "native MoE uses 1/8 of experts and is fine"
intuition: that survival is TRAINED, not intrinsic. Post-hoc reductions of native
top-K hurt -- the right reference for our post-hoc G-MoEfication numbers.

OLMoE was trained with top-K=8 of 64 experts per token. We measure prediction divergence
vs the native K=8 baseline when we force K' < 8 at inference (same weights, no retrain,
router unchanged except K). If K'=4 already degrades sharply, then "12.5% sparsity is
free" is a property of the TRAINED selection regime, not of MoE redundancy in general --
which is exactly why post-hoc MoEfication of a dense FFN (exp 107-111) is hard at 50%.

Run: python3 experiments/112_native_topk_reduction.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_TOK = 1024
CHUNK = 256
KS = [8, 6, 4, 2, 1]


def cp_upper(k, n, conf=0.95):
    try:
        from scipy.stats import beta
        return 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))
    except Exception:
        return min(1.0, k / n + math.sqrt(math.log(1.0 / (1.0 - conf)) / (2 * n)))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers
    native = layers[0].mlp.top_k

    def set_K(k):
        for L in layers:
            L.mlp.top_k = k

    @torch.no_grad()
    def preds(k):
        set_K(k)
        out = []
        for c0 in range(0, N_TOK, CHUNK):
            x = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            out.append(model(x).logits[0, :-1].float().argmax(-1).cpu())
        return out

    bp = preds(native)
    print(f"native top-K = {native}; reference predictions from this setting")
    print(f"\n  {'K':>3s} | {'%mem':>5s} | {'emp pred-diff':>13s} | "
          f"{'cert <= (95%)':>14s}")
    print("  " + "-" * 50)
    for k in KS:
        ps = preds(k)
        kbad = tot = 0
        for a, b in zip(bp, ps):
            kbad += int((a != b).sum().item())
            tot += a.numel()
        mem = k / native * 100
        print(f"  {k:>3d} | {mem:>4.0f}% | {kbad/tot*100:>12.2f}% | "
              f"{cp_upper(kbad, tot)*100:>13.2f}%", flush=True)

    print("\nNative trained top-K=8 vs forced K'<8 with the same weights/router.")
    print("If divergence grows fast with K -> 'MoE uses 1/8 freely' is a TRAINED")
    print("property of the selected sparsity level, not generic MoE redundancy.")
    print("This is the right reference for the post-hoc G-MoEfication numbers in")
    print("exp 107-111 (post-hoc decomposition of a dense FFN, also untrained-for).")


if __name__ == "__main__":
    main()
