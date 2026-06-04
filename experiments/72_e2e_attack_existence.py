"""Experiment 72 — Does an ATTACK exist? The value-of-the-guarantee test.

The user's point: a worst-case guarantee is only meaningful if, WITHOUT it, an
adversarial counterexample always exists. If no input can make the quantized model
diverge meaningfully from fp16, the certificate guards against nothing -> vacuous.

We attack the deployed quantized model directly at the MODEL OUTPUT level. For a few
real prompts we PGD-perturb the INPUT EMBEDDINGS within an L2 ball to MAXIMIZE the
divergence between the fp16 model and the quantized model's next-token logits:
    max_{||x-x0||_2 <= eps}  || Y_fp16(x) - Y_quant(x) ||_2 .
We report, per prompt and on average:
  - clean divergence ||Y_f(x0) - Y_q(x0)||           (what testing on x0 sees)
  - ADVERSARIAL divergence at the PGD optimum         (the worst case)
  - RANDOM-perturbation divergence at the same norm   (control: is the attack real?)
  - next-token argmax agreement fp16-vs-quant: clean / adversarial / random
  - KL(p_f || p_q): clean / adversarial.

Reading:
  adv >> clean AND argmax flips under adv but not clean  -> a counterexample exists,
     the lack of a guarantee is exploitable, verification's premise holds.
  adv ~ clean ~ random, no argmax flips -> NO counterexample (MoE redundancy absorbs
     the quant error), the guarantee is vacuous; consistent with exp 66 / 71.

Memory: fp16 model (~19GB) is resident; quantized EXPERT weights live on CPU and are
swapped in per forward (router/attn/embed unchanged by quant). The fp16 branch is run
under no_grad (so the in-place weight swap never corrupts an autograd graph); the PGD
gradient flows through the quant branch only (a standard ascent direction; the final
divergence is measured exactly with both models at the perturbed point).

GPU, live OLMoE. Run: python3 experiments/72_e2e_attack_existence.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
BITS_LIST = [8, 4]        # 8-bit = near-lossless "validated" artifact; 4-bit = already-lossy ref
PGD_STEPS = 30
EPS_FRACS = [0.02, 0.05, 0.10]   # eps as fraction of mean ||embedding||_2
SEQ = 32
PROMPTS = [
    "The capital of France is",
    "def add(a, b):\n    return",
    "Once upon a time, there was a",
    "The mitochondria is the powerhouse of the",
    "2 + 2 = ",
    "To be or not to be, that is the",
]


def quant(W, bits):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.92, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # collect expert weight params; store fp16 (CPU) and quant (CPU)
    expert_params = []   # (param_tensor_on_gpu, name)
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue
        for e in range(mlp.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                w = getattr(mlp.experts[e], proj).weight
                expert_params.append(w)
    print(f"  {len(expert_params)} expert weight matrices", flush=True)
    fp16_cpu = [w.detach().to("cpu").clone() for w in expert_params]
    print("  fp16 expert weights cached on CPU", flush=True)

    @torch.no_grad()
    def set_state(state_cpu):
        for w, s in zip(expert_params, state_cpu):
            w.data.copy_(s.to(dev, non_blocking=True))
        if dev == "cuda":
            torch.cuda.synchronize()

    emb = model.get_input_embeddings()

    @torch.no_grad()
    def probs(x):
        """next-token probability distribution (float32, bounded objective space)."""
        return model(inputs_embeds=x).logits[:, -1, :].float().softmax(-1)

    def tv(pa, pb):
        return 0.5 * (pa - pb).abs().sum().item()

    import statistics as st
    for BITS in BITS_LIST:
        quant_cpu = [quant(w.detach().float(), BITS).to(torch.float16).cpu()
                     for w in fp16_cpu]
        print(f"\n{'='*92}\n  E2E attack ({BITS}-bit quant): maximize TV(p_fp16(x), "
              f"p_quant(x)) over L2 input-embed ball\n{'='*92}")
        print(f"  {'eps_frac':>8s} | {'clean TV':>9s} | {'ADV TV':>9s} | {'rand TV':>8s} | "
              f"{'adv/clean':>9s} | {'argmax flip c/a/r':>17s} | {'KL clean/adv':>14s}")
        print("  " + "-"*90)

        for ef in EPS_FRACS:
            clean_d, adv_d, rand_d, kl_c, kl_a = [], [], [], [], []
            flips_c = flips_a = flips_r = 0
            n = 0
            for prompt in PROMPTS:
                ids = tok(prompt, return_tensors="pt").input_ids[:, :SEQ].to(dev)
                with torch.no_grad():
                    x0 = emb(ids).detach()
                eps = ef * x0.norm(dim=-1).mean().item() * (x0.shape[1] ** 0.5)

                set_state(fp16_cpu); pf0 = probs(x0)
                set_state(quant_cpu); pq0 = probs(x0)
                clean_d.append(tv(pf0, pq0))
                flips_c += int(pf0.argmax(-1).item() != pq0.argmax(-1).item())
                kl_c.append(F.kl_div(pq0.clamp_min(1e-12).log(), pf0,
                                     reduction="sum").item())

                # PGD: maximize ||softmax(pq) - softmax(pf)||^2 (bounded, stable)
                delta = torch.zeros_like(x0).requires_grad_(True)
                for _ in range(PGD_STEPS):
                    x = x0 + delta
                    set_state(fp16_cpu)
                    with torch.no_grad():
                        pf = model(inputs_embeds=x).logits[:, -1, :].float()
                        pf = pf.softmax(-1).detach()
                    set_state(quant_cpu)
                    pqlog = model(inputs_embeds=x).logits[:, -1, :].float()
                    pq = pqlog.softmax(-1)
                    loss = ((pq - pf) ** 2).sum()
                    g, = torch.autograd.grad(loss, delta)
                    if not torch.isfinite(g).all():
                        break
                    with torch.no_grad():
                        delta += (eps / 5) * g / (g.norm() + 1e-12)
                        nrm = delta.norm()
                        if nrm > eps:
                            delta.mul_(eps / nrm)
                    delta = delta.detach().requires_grad_(True)

                with torch.no_grad():
                    xa = x0 + delta.detach()
                    set_state(fp16_cpu); pfa = probs(xa)
                    set_state(quant_cpu); pqa = probs(xa)
                    adv_d.append(tv(pfa, pqa))
                    flips_a += int(pfa.argmax(-1).item() != pqa.argmax(-1).item())
                    kl_a.append(F.kl_div(pqa.clamp_min(1e-12).log(), pfa,
                                         reduction="sum").item())

                    r = torch.randn_like(x0); r = r * (eps / r.norm())
                    set_state(fp16_cpu); pfr = probs(x0 + r)
                    set_state(quant_cpu); pqr = probs(x0 + r)
                    rand_d.append(tv(pfr, pqr))
                    flips_r += int(pfr.argmax(-1).item() != pqr.argmax(-1).item())
                n += 1

            print(f"  {ef:>8.2f} | {st.mean(clean_d):>9.4f} | {st.mean(adv_d):>9.4f} | "
                  f"{st.mean(rand_d):>8.4f} | "
                  f"{st.mean(adv_d)/max(st.mean(clean_d),1e-9):>8.2f}x | "
                  f"{flips_c:>3d}/{flips_a:>3d}/{flips_r:<3d} (of {n}) | "
                  f"{st.mean(kl_c):>6.4f}/{st.mean(kl_a):<6.4f}", flush=True)

    print("\n  Reading: adv TV >> clean AND argmax flips under adv (not clean/rand)")
    print("  => a counterexample exists, the guarantee guards something real.")
    print("  adv ~ clean ~ rand and no flips => no exploitable counterexample; the")
    print("  guarantee is vacuous for MoE quant (redundancy absorbs it; cf. exp 66/71).")


if __name__ == "__main__":
    main()
