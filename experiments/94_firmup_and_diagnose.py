"""Experiment 94 — Firm up exp 92/93 (multi-chunk eval, less noise) and DIAGNOSE the
code exception.

(1) FIRM UP: evaluate held-out next-token top-1 accuracy over 1024 tokens split into
    4 chunks -> report mean +/- std for CERT-order vs FREQ-order pruning across the
    compression sweep, per domain (code, prose).
(2) DIAGNOSE: the certified contribution criterion only beats frequency when the two
    orderings DIFFER. We measure, per domain, the Spearman rank correlation between
    expert frequency and expert contribution (g*||E||) -- if HIGH (code?), the most-used
    experts ARE the high-impact ones, so cert-order ~ freq-order and there is no
    advantage; if LOW (prose?), they differ and contribution-ordering wins.

GPU, live OLMoE. Run: python3 experiments/94_firmup_and_diagnose.py
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
KGRID = [24, 32, 40, 48, 56]


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


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm() + 1e-12)).item()


@torch.no_grad()
def cert_contrib_freq(mlp, H, eps, dev):
    Wg = mlp.gate.weight.float(); rown = Wg.norm(dim=1)
    lg = H @ Wg.T
    ub = lg + eps * rown; lb = lg - eps * rown
    K, N = mlp.top_k, mlp.num_experts
    dead = [e for e in range(N) if ((lb > ub[:, e:e+1]).sum(1) >= K).all()]
    topv, topi = lg.topk(K, dim=-1); g = torch.softmax(topv, dim=-1)
    contrib = torch.zeros(N, device=dev); freq = torch.zeros(N, device=dev)
    for e in range(N):
        sel = (topi == e)
        freq[e] = sel.any(1).sum()
        if sel.any():
            ti = sel.any(1).nonzero().flatten()
            W1 = mlp.experts[e].gate_proj.weight.float()
            W2 = mlp.experts[e].up_proj.weight.float()
            W3 = mlp.experts[e].down_proj.weight.float()
            Ee = (F.silu(H[ti] @ W1.T) * (H[ti] @ W2.T)) @ W3.T
            contrib[e] = (g[sel] * Ee.norm(dim=-1)).mean()
    return set(dead), contrib, freq


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
        cert_order, freq_order, sp = {}, {}, []
        for li in range(nL):
            mlp = layers[li].mlp
            dead, contrib, freq = cert_contrib_freq(mlp, Hc[li], EPS, dev)
            key = [(0 if e in dead else 1, contrib[e].item()) for e in range(Nexp)]
            cert_order[li] = sorted(range(Nexp), key=lambda e: key[e])
            freq_order[li] = torch.argsort(freq).tolist()
            sp.append(spearman(freq, contrib))
        sp_mean = sum(sp) / len(sp)

        @torch.no_grad()
        def acc_chunks():
            accs = []
            for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
                x = ids[c0:c0 + CHUNK].unsqueeze(0)
                p = model(x).logits[0, :-1].float().argmax(-1)
                accs.append((p == x[0, 1:]).float().mean().item() * 100)
            return accs

        base = acc_chunks()
        print(f"\n{'='*70}\n  DOMAIN={name}: held-out top-1 acc (mean+/-std over "
              f"{N_EVAL//CHUNK} chunks)\n  baseline={st.mean(base):.1f}; "
              f"freq<->contribution Spearman={sp_mean:+.2f}\n{'='*70}")
        print(f"  {'%pruned':>7s} | {'CERT-order':>14s} | {'FREQ-order':>14s} | {'gap':>6s}")
        print("  " + "-"*52)
        for k in KGRID:
            prune(cert_order, k); ac = acc_chunks(); restore()
            prune(freq_order, k); af = acc_chunks(); restore()
            print(f"  {k/Nexp*100:>6.0f}% | {st.mean(ac):>6.1f}+/-{st.pstdev(ac):>4.1f} | "
                  f"{st.mean(af):>6.1f}+/-{st.pstdev(af):>4.1f} | "
                  f"{st.mean(ac)-st.mean(af):>+6.1f}", flush=True)

    print(f"\n  HIGH freq<->contribution Spearman => orderings coincide => no cert advantage")
    print("  (the code case?); LOW => they differ => contribution-ordering wins (prose).")
    print("  Gaps now mean+/-std over chunks: real if gap >> std.")


if __name__ == "__main__":
    main()
