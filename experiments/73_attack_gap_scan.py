"""Experiment 73 — The "reversal key" test: scan bit-width to find a regime where
CLEAN looks fine (testing passes) but ADVERSARIAL breaks it (counterexample exists).

The guarantee is meaningful only at a precision where:
  clean divergence is SMALL  (the quantization PASSES validation -> would be shipped)
  AND adversarial divergence is LARGE (a counterexample testing missed)
  AND adv >> random (it is a real attack, not perturbation noise).
exp 72: 8-bit has NO gap (adv<=clean~rand); 4-bit already diverges on CLEAN (TV 0.25,
testing rejects -> moot). Open question (user): is there an INTERMEDIATE precision
(7/6/5-bit) where clean is acceptable but an adversarial input opens a large gap?

EFFICIENCY: per-step fp16<->quant weight swaps are the bottleneck. We avoid them:
fp16 CLEAN reference distributions pf0 are precomputed once; for each bit-width the
quant model stays resident while PGD pushes the quant next-token distribution AWAY
from the trusted fp16 clean answer (max ||p_quant(x) - pf0||^2, a deployment attack);
then ONE fp16 pass measures the TRUE divergence ||p_fp16(x_adv) - p_quant(x_adv)||.
~2 weight swaps per bit instead of thousands.

GPU, live OLMoE. Run: python3 experiments/73_attack_gap_scan.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
BITS_LIST = [8, 7, 6, 5, 4, 3]
PGD_STEPS = 40
EPS_FRACS = [0.03, 0.06]
SEQ = 32
PROMPTS = [
    "The capital of France is",
    "def add(a, b):\n    return",
    "Once upon a time, there was a",
    "The mitochondria is the powerhouse of the",
    "2 + 2 = ",
    "To be or not to be, that is the",
    "The quick brown fox jumps over the",
    "import numpy as np\nx =",
]


def quant(W, bits):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import statistics as st
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.92, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    expert_params = []
    for layer in model.model.layers:
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue
        for e in range(mlp.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                expert_params.append(getattr(mlp.experts[e], proj).weight)
    fp16_cpu = [w.detach().to("cpu").clone() for w in expert_params]
    print(f"  {len(expert_params)} expert matrices cached", flush=True)

    @torch.no_grad()
    def set_state(state):
        for w, s in zip(expert_params, state):
            w.data.copy_(s.to(dev, non_blocking=True))
        if dev == "cuda":
            torch.cuda.synchronize()

    emb = model.get_input_embeddings()

    @torch.no_grad()
    def probs_at(x):
        return model(inputs_embeds=x).logits[:, -1, :].float().softmax(-1)

    def tv(a, b):
        return 0.5 * (a - b).abs().sum().item()

    # embeddings + eps per prompt
    X0, EPS = [], {}
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids[:, :SEQ].to(dev)
        with torch.no_grad():
            x0 = emb(ids).detach()
        X0.append(x0)
        for ef in EPS_FRACS:
            EPS[(prompt, ef)] = ef * x0.norm(dim=-1).mean().item() * (x0.shape[1] ** 0.5)

    # fp16 clean reference distributions (one fp16 pass)
    set_state(fp16_cpu)
    pf0 = [probs_at(x0) for x0 in X0]

    print(f"\n{'='*96}\n  Attack-gap scan: clean vs ADVERSARIAL vs random next-token TV "
          f"(fp16 vs quant)\n{'='*96}")
    print(f"  {'bits':>4s} {'eps':>5s} | {'clean TV':>9s} | {'ADV TV':>9s} | {'rand TV':>8s}"
          f" | {'adv/clean':>9s} | {'adv/rand':>8s} | {'flip c/a/r':>11s}")
    print("  " + "-"*94)

    for BITS in BITS_LIST:
        quant_cpu = [quant(w.float(), BITS).to(torch.float16) for w in fp16_cpu]
        for ef in EPS_FRACS:
            # PGD with ONLY the quant model resident: push p_quant(x) away from pf0
            set_state(quant_cpu)
            pq0, x_adv, x_rnd = [], [], []
            for i, prompt in enumerate(PROMPTS):
                x0 = X0[i]; eps = EPS[(prompt, ef)]
                pq0.append(probs_at(x0))
                delta = torch.zeros_like(x0).requires_grad_(True)
                for _ in range(PGD_STEPS):
                    pq = model(inputs_embeds=x0 + delta).logits[:, -1, :].float().softmax(-1)
                    loss = ((pq - pf0[i]) ** 2).sum()
                    if not torch.isfinite(loss):
                        break
                    g, = torch.autograd.grad(loss, delta)
                    if not torch.isfinite(g).all():
                        break
                    with torch.no_grad():
                        delta += (eps / 6) * g / (g.norm() + 1e-12)
                        nrm = delta.norm()
                        if nrm > eps:
                            delta.mul_(eps / nrm)
                    delta = delta.detach().requires_grad_(True)
                x_adv.append((x0 + delta.detach()))
                r = torch.randn_like(x0); r = r * (eps / r.norm())
                x_rnd.append(x0 + r)
            with torch.no_grad():
                pqa = [probs_at(x) for x in x_adv]
                pqr = [probs_at(x) for x in x_rnd]
            # ONE fp16 pass: true fp16 distributions at clean/adv/rand points
            set_state(fp16_cpu)
            with torch.no_grad():
                pfa = [probs_at(x) for x in x_adv]
                pfr = [probs_at(x) for x in x_rnd]
            cd = [tv(pf0[i], pq0[i]) for i in range(len(PROMPTS))]
            ad = [tv(pfa[i], pqa[i]) for i in range(len(PROMPTS))]
            rd = [tv(pfr[i], pqr[i]) for i in range(len(PROMPTS))]
            fc = sum(int(pf0[i].argmax(-1).item() != pq0[i].argmax(-1).item())
                     for i in range(len(PROMPTS)))
            fa = sum(int(pfa[i].argmax(-1).item() != pqa[i].argmax(-1).item())
                     for i in range(len(PROMPTS)))
            fr = sum(int(pfr[i].argmax(-1).item() != pqr[i].argmax(-1).item())
                     for i in range(len(PROMPTS)))
            mc, ma, mr = st.mean(cd), st.mean(ad), st.mean(rd)
            print(f"  {BITS:>4d} {ef:>5.2f} | {mc:>9.4f} | {ma:>9.4f} | {mr:>8.4f} | "
                  f"{ma/max(mc,1e-9):>8.2f}x | {ma/max(mr,1e-9):>7.2f}x | "
                  f"{fc:>2d}/{fa:>2d}/{fr:<2d}", flush=True)
        print("  " + "-"*94, flush=True)

    print("\n  REVERSAL if some row has small clean TV AND adv/clean >> 1 AND adv/rand >> 1")
    print("  (testing passes, adversary breaks it, real attack). Otherwise the gap only")
    print("  opens where clean is already large -> testing already rejects -> no reversal.")


if __name__ == "__main__":
    main()
