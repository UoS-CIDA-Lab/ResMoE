"""Experiment 93 — Does the certified/contribution selection criterion beat frequency
selection EARLIER (lower compression) on HARDER tasks than the trivial arithmetic
template (exp 92, gap only at 88%)?

Harder domains have less redundancy headroom, so the gap between the certified-derived
ordering (drop provably-dead first, then by ascending contribution = impact) and the
frequency ordering (drop least-used) should open at lower compression. We rerun the
exp-92 sweep on CODE (HumanEval) and PROSE (WikiText): calibrate + held-out evaluate on
the same domain, comparing held-out next-token top-1 accuracy of CERT-order vs FREQ-order
pruning across per-layer drop counts.

GPU, live OLMoE. Run: python3 experiments/93_harder_task_sweep.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 128
N_EVAL = 256
EPS = 0.05
KGRID = [16, 24, 32, 40, 48, 56]


def domains(tok):
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    he = load_dataset("openai_humaneval", split="test")
    prose = "\n\n".join(t for t in wt["text"] if t.strip())
    code = "\n\n".join(he["prompt"])
    return {"code": tok(code, return_tensors="pt").input_ids[0],
            "prose": tok(prose, return_tensors="pt").input_ids[0]}


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
def cert_and_contrib(mlp, H, eps, dev):
    Wg = mlp.gate.weight.float(); rown = Wg.norm(dim=1)
    lg = H @ Wg.T
    ub = lg + eps * rown; lb = lg - eps * rown
    K, N = mlp.top_k, mlp.num_experts
    dead = [e for e in range(N) if ((lb > ub[:, e:e+1]).sum(1) >= K).all()]
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
    return set(dead), contrib


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
    nL = len(layers)
    Nexp = layers[0].mlp.num_experts

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

    for name, ids in doms.items():
        ids = ids.to(dev)
        Hc = collect(model, ids, dev, list(range(nL)), 0, N_CALIB)
        cert_order, freq_order, budget = {}, {}, []
        for li in range(nL):
            mlp = layers[li].mlp
            dead, contrib = cert_and_contrib(mlp, Hc[li], EPS, dev)
            budget.append(len(dead))
            key = [(0 if e in dead else 1, contrib[e].item()) for e in range(Nexp)]
            cert_order[li] = sorted(range(Nexp), key=lambda e: key[e])
            tk = (Hc[li] @ mlp.gate.weight.float().T).topk(mlp.top_k, -1).indices
            freq = torch.bincount(tk.flatten(), minlength=Nexp)
            freq_order[li] = torch.argsort(freq).tolist()
        avg_b = sum(budget) / nL

        @torch.no_grad()
        def acc():
            x = ids[N_CALIB:N_CALIB + N_EVAL].unsqueeze(0)
            p = model(x).logits[0, :-1].float().argmax(-1)
            return (p == x[0, 1:]).float().mean().item() * 100

        base = acc()
        print(f"\n{'='*66}\n  DOMAIN={name}: held-out next-token top-1 acc (%) vs compression"
              f"\n  baseline={base:.1f}; certified budget/layer={avg_b:.0f}/{Nexp}\n{'='*66}")
        print(f"  {'k/layer':>7s} | {'%pruned':>7s} | {'CERT-order':>10s} | "
              f"{'FREQ-order':>10s} | {'cert-freq':>9s}")
        print("  " + "-"*56)
        for k in KGRID:
            prune(cert_order, k); ac = acc(); restore()
            prune(freq_order, k); af = acc(); restore()
            print(f"  {k:>7d} | {k/Nexp*100:>6.0f}% | {ac:>10.1f} | {af:>10.1f} | "
                  f"{ac-af:>+8.1f}", flush=True)

    print(f"\n  Gap (cert-freq) opening at LOWER k on harder domains => the certified")
    print("  contribution criterion is a better selection where redundancy is scarcer.")


if __name__ == "__main__":
    main()
