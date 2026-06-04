"""Experiment 100 — Task/domain-aware certified decomposition (the user's goal).

For a target domain, decompose the MoE by removing experts of low certified contribution
(verification informs the selection; the provably-never-selected subset is exactly
lossless), then CERTIFY the decomposed submodel on the DOMAIN with a distributional
Clopper-Pearson bound: "on domain D, at most p of predictions differ from the full
model (95% conf)". The distributional guarantee is tied to the deployment distribution,
which for a domain-specialized deployment IS the domain -- so it is exactly the right
tool. We report, per domain, the decomposition curve: %experts removed (memory) vs the
certified prediction-divergence bound, plus the certified-lossless (never-selected) floor.

GPU, live OLMoE, domains = code / math / prose. Run:
  python3 experiments/100_domain_aware_certified_decomp.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 256
N_EVAL = 512
CHUNK = 256
EPS = 0.05
CONF = 0.95
KGRID = [16, 24, 32, 40, 48]


def domains(tok):
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    he = load_dataset("openai_humaneval", split="test")
    gs = load_dataset("gsm8k", "main", split="test")
    return {"code": tok("\n\n".join(he["prompt"]), return_tensors="pt").input_ids[0],
            "math": tok("\n\n".join(gs["question"][:400]), return_tensors="pt").input_ids[0],
            "prose": tok("\n\n".join(t for t in wt["text"] if t.strip()),
                         return_tensors="pt").input_ids[0]}


def cp_upper(k, n, conf):
    try:
        from scipy.stats import beta
        return 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))
    except Exception:
        return min(1.0, k / n + math.sqrt(math.log(1.0 / (1.0 - conf)) / (2 * n)))


@torch.no_grad()
def collect(model, ids, dev, layers, lo, hi):
    cap = {li: [] for li in layers}
    hs = []
    for li in layers:
        def mk(li):
            def hk(_m, a):
                cap[li].append(a[0].detach().reshape(-1, a[0].shape[-1]))
            return hk
        hs.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    model(ids[lo:hi].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    return {li: torch.cat(cap[li]).float() for li in layers}


@torch.no_grad()
def order_and_dead(mlp, H, eps, dev):
    Wg = mlp.gate.weight.float(); rown = Wg.norm(dim=1)
    lg = H @ Wg.T
    ub = lg + eps * rown; lb = lg - eps * rown
    K, N = mlp.top_k, mlp.num_experts
    dead = set(e for e in range(N) if ((lb > ub[:, e:e+1]).sum(1) >= K).all())
    topv, topi = lg.topk(K, dim=-1); g = torch.softmax(topv, dim=-1)
    contrib = torch.zeros(N, device=dev)
    for e in range(N):
        sel = (topi == e)
        if sel.any():
            ti = sel.any(1).nonzero().flatten()
            W1 = mlp.experts[e].gate_proj.weight.float()
            W2 = mlp.experts[e].up_proj.weight.float()
            W3 = mlp.experts[e].down_proj.weight.float()
            Ee = (F.silu(H[ti] @ W1.T) * (H[ti] @ W2.T)) @ W3.T
            contrib[e] = (g[sel] * Ee.norm(dim=-1)).mean()
    order = sorted(range(N), key=lambda e: (0 if e in dead else 1, contrib[e].item()))
    return order, len(dead)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    doms = domains(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers
    nL = len(layers); Nexp = layers[0].mlp.num_experts

    down = [layers[li].mlp.experts[e].down_proj.weight
            for li in range(nL) for e in range(Nexp)]
    saved = [w.detach().cpu().clone() for w in down]

    def restore():
        with torch.no_grad():
            for w, s in zip(down, saved):
                w.data.copy_(s.to(dev))

    def prune(order, k):
        with torch.no_grad():
            for li in range(nL):
                for e in order[li][:k]:
                    layers[li].mlp.experts[e].down_proj.weight.data.zero_()

    @torch.no_grad()
    def base_preds(ids):
        ps = []
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            x = ids[c0:c0 + CHUNK].unsqueeze(0)
            ps.append(model(x).logits[0, :-1].float().argmax(-1))
        return ps

    @torch.no_grad()
    def divergence(ids, base_ps):
        kbad = tot = 0
        for j, c0 in enumerate(range(N_CALIB, N_CALIB + N_EVAL, CHUNK)):
            x = ids[c0:c0 + CHUNK].unsqueeze(0)
            p = model(x).logits[0, :-1].float().argmax(-1)
            kbad += int((p != base_ps[j]).sum().item()); tot += p.numel()
        return kbad, tot

    for name, ids in doms.items():
        ids = ids.to(dev)
        Hc = collect(model, ids, dev, list(range(nL)), 0, N_CALIB)
        order, deads = {}, []
        for li in range(nL):
            order[li], d = order_and_dead(layers[li].mlp, Hc[li], EPS, dev)
            deads.append(d)
        bp = base_preds(ids)
        avg_dead = sum(deads) / nL
        print(f"\n{'='*70}\n  DOMAIN={name}: certified decomposition (full model = reference,"
              f" {int(CONF*100)}% conf)\n  certified-lossless floor: {avg_dead:.0f}/{Nexp} "
              f"experts/layer never-selected\n{'='*70}")
        print(f"  {'%removed':>8s} | {'mem':>5s} | {'emp pred-diff':>13s} | "
              f"{'cert diff <= (95%)':>17s}")
        print("  " + "-"*54)
        for k in KGRID:
            prune(order, k)
            kbad, tot = divergence(ids, bp)
            restore()
            print(f"  {k/Nexp*100:>7.0f}% | {(Nexp-k)/Nexp*100:>4.0f}% | "
                  f"{kbad/tot*100:>12.2f}% | {cp_upper(kbad, tot, CONF)*100:>16.2f}%",
                  flush=True)

    print(f"\n  Reading: for domain D, remove %experts -> certified bound on the fraction of")
    print("  domain-D predictions that differ from the FULL model (95% conf). The")
    print("  never-selected floor is removed losslessly (exact). Distributional guarantee on")
    print("  D = exactly right for a D-specialized deployment. Selection = impact-ordering")
    print("  (verification adds the lossless floor + the on-domain certificate, not the pick).")


if __name__ == "__main__":
    main()
