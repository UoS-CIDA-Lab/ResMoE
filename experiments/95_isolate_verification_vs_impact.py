"""Experiment 95 — Control: is the task-accuracy advantage (exp 92-94) from VERIFICATION
or just from IMPACT-based selection (which beats frequency without any verification)?

Three pruning orderings at matched count, held-out next-token accuracy (multi-chunk):
  FREQ           : drop least-FREQUENT experts (weak baseline).
  IMPACT-emp     : drop by ascending empirical contribution g*||E|| (NO verification,
                   NO provably-dead-first) -- a standard impact/magnitude pruner.
  CERT           : drop provably-dead first, then ascending contribution
                   (verification-derived ordering).
Prediction: CERT ~ IMPACT-emp >> FREQ. If so, the performance win is the IMPACT
criterion (a known effect vs the weak frequency baseline), and verification's UNIQUE
contribution is the GUARANTEE (provably-dead lossless + delta-bounded layer), not the
accuracy. If CERT > IMPACT-emp, the dead-first/verification ordering adds accuracy too.

GPU, live OLMoE, code + prose. Run: python3 experiments/95_isolate_verification_vs_impact.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 256
N_EVAL = 1024
CHUNK = 256
EPS = 0.05
KGRID = [32, 40, 48, 56]


def domains(tok):
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    he = load_dataset("openai_humaneval", split="test")
    return {"code": tok("\n\n".join(he["prompt"]), return_tensors="pt").input_ids[0],
            "prose": tok("\n\n".join(t for t in wt["text"] if t.strip()),
                         return_tensors="pt").input_ids[0]}


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
def orders(mlp, H, eps, dev):
    Wg = mlp.gate.weight.float(); rown = Wg.norm(dim=1)
    lg = H @ Wg.T
    ub = lg + eps * rown; lb = lg - eps * rown
    K, N = mlp.top_k, mlp.num_experts
    dead = set(e for e in range(N) if ((lb > ub[:, e:e+1]).sum(1) >= K).all())
    topv, topi = lg.topk(K, dim=-1); g = torch.softmax(topv, dim=-1)
    contrib = torch.zeros(N, device=dev); freq = torch.zeros(N, device=dev)
    for e in range(N):
        sel = (topi == e); freq[e] = sel.any(1).sum()
        if sel.any():
            ti = sel.any(1).nonzero().flatten()
            W1 = mlp.experts[e].gate_proj.weight.float()
            W2 = mlp.experts[e].up_proj.weight.float()
            W3 = mlp.experts[e].down_proj.weight.float()
            Ee = (F.silu(H[ti] @ W1.T) * (H[ti] @ W2.T)) @ W3.T
            contrib[e] = (g[sel] * Ee.norm(dim=-1)).mean()
    freq_o = torch.argsort(freq).tolist()
    impact_o = torch.argsort(contrib).tolist()
    cert_o = sorted(range(N), key=lambda e: (0 if e in dead else 1, contrib[e].item()))
    return freq_o, impact_o, cert_o


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import statistics as st
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

    for name, ids in doms.items():
        ids = ids.to(dev)
        Hc = collect(model, ids, dev, list(range(nL)), 0, N_CALIB)
        fo, io, co = {}, {}, {}
        for li in range(nL):
            fo[li], io[li], co[li] = orders(layers[li].mlp, Hc[li], EPS, dev)

        @torch.no_grad()
        def acc():
            a = []
            for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
                x = ids[c0:c0 + CHUNK].unsqueeze(0)
                p = model(x).logits[0, :-1].float().argmax(-1)
                a.append((p == x[0, 1:]).float().mean().item() * 100)
            return st.mean(a)

        base = acc()
        print(f"\n{'='*70}\n  DOMAIN={name}: held-out top-1 acc (%), baseline={base:.1f}"
              f"\n{'='*70}")
        print(f"  {'%pruned':>7s} | {'FREQ':>6s} | {'IMPACT-emp':>10s} | {'CERT':>6s} | "
              f"{'CERT-IMPACT':>11s} | {'IMPACT-FREQ':>11s}")
        print("  " + "-"*64)
        for k in KGRID:
            prune(fo, k); af = acc(); restore()
            prune(io, k); ai = acc(); restore()
            prune(co, k); ac = acc(); restore()
            print(f"  {k/Nexp*100:>6.0f}% | {af:>6.1f} | {ai:>10.1f} | {ac:>6.1f} | "
                  f"{ac-ai:>+11.1f} | {ai-af:>+11.1f}", flush=True)

    print(f"\n  CERT ~ IMPACT-emp (CERT-IMPACT ~ 0) and both >> FREQ (IMPACT-FREQ > 0)")
    print("  => the accuracy win is the IMPACT criterion, NOT verification; verification's")
    print("  unique contribution is the GUARANTEE (provably-dead + delta-bounded), not perf.")
    print("  If CERT-IMPACT > 0, the verification-derived dead-first ordering adds accuracy too.")


if __name__ == "__main__":
    main()
