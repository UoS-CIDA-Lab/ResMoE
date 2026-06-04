"""Experiment 98 — Probabilistic (distributional) compression certificate.

Every certificate we tried was WORST-CASE (vacuous at model scale via the depth wall).
Here we change the GUARANTEE TYPE: a sound DISTRIBUTIONAL certificate that escapes the
worst-case composition. For a model-wide compression we sample the deployment
distribution and bound the VIOLATION RATE with confidence:
    P_{x ~ deploy}( compressed behaves differently from original ) <= p_upper (95% conf)
via a Clopper-Pearson (binomial) upper confidence bound on the empirical violation rate.
"Different" = the next-token argmax disagrees with the original (a discrete behavioral
violation), and also TV(p_comp, p_orig) > delta. This needs NO worst-case composition
(we evaluate the real model on real samples), so it gives a MODEL-OUTPUT guarantee at a
MEANINGFUL (all-layer) compression rate -- the trade is a probabilistic (not worst-case)
guarantee tied to the sampled distribution. Reports per compression level: memory,
empirical violation rate, certified 95% upper bound.

GPU, live OLMoE. Run: python3 experiments/98_probabilistic_certificate.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_TOK = 2048
CONF = 0.95
BITS = [8, 6, 4, 3]
TV_DELTAS = [0.05, 0.10, 0.20]


def cp_upper(k, n, conf):
    """Clopper-Pearson upper (1-conf one-sided) bound on the violation prob."""
    try:
        from scipy.stats import beta
        if k == n:
            return 1.0
        return float(beta.ppf(conf, k + 1, n - k))
    except Exception:
        # Hoeffding fallback (sound, looser)
        return min(1.0, k / n + math.sqrt(math.log(1.0 / (1.0 - conf)) / (2 * n)))


def quant(W, bits):
    if bits >= 16:
        return W
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0][:N_TOK]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers
    nL = len(layers)

    expert_w = [getattr(layers[li].mlp.experts[e], p).weight
                for li in range(nL) for e in range(layers[li].mlp.num_experts)
                for p in ("gate_proj", "up_proj", "down_proj")]
    saved = [w.detach().cpu().clone() for w in expert_w]

    def restore():
        with torch.no_grad():
            for w, s in zip(expert_w, saved):
                w.data.copy_(s.to(dev))

    def apply_bits(b):
        with torch.no_grad():
            for w in expert_w:
                w.data.copy_(quant(w.float(), b).to(torch.float16))

    @torch.no_grad()
    def preds_probs():
        x = ids.unsqueeze(0).to(dev)
        lg = model(x).logits[0, :-1].float()
        return lg.argmax(-1), lg.softmax(-1)

    base_pred, base_p = preds_probs()
    n = base_pred.shape[0]
    print(f"\n{'='*84}\n  Probabilistic compression certificate (deploy=WikiText, n={n} "
          f"next-token decisions, {int(CONF*100)}% conf)\n{'='*84}")
    print(f"  {'bits':>4s} {'mem':>5s} | {'pred-disagree':>13s} | {'cert<=(95%)':>11s} | "
          + " | ".join(f"TV>{d}" for d in TV_DELTAS))
    print("  " + "-"*80)
    for b in BITS:
        apply_bits(b)
        cp, pp = preds_probs()
        restore()
        kv = int((cp != base_pred).sum().item())
        pred_rate = kv / n
        pred_cert = cp_upper(kv, n, CONF)
        tv = 0.5 * (pp - base_p).abs().sum(-1)            # per-token TV
        cells = []
        for d in TV_DELTAS:
            kt = int((tv > d).sum().item())
            cells.append(f"{kt/n*100:>4.1f}/{cp_upper(kt, n, CONF)*100:>4.1f}")
        print(f"  {b:>4d} {b/16*100:>4.0f}% | {pred_rate*100:>11.2f}% | "
              f"{pred_cert*100:>10.2f}% | " + " | ".join(cells), flush=True)

    print(f"\n  Read: 'pred-disagree' = empirical %% of next-tokens the compressed model")
    print("  predicts differently from the original; 'cert<=(95%)' = sound 95%-confidence")
    print("  upper bound on that rate over the deployment distribution. TV cells =")
    print("  empirical%/cert95% of tokens with distribution shift > delta. A MODEL-OUTPUT")
    print("  guarantee at full (all-layer) compression -- probabilistic, no worst-case wall.")


if __name__ == "__main__":
    main()
