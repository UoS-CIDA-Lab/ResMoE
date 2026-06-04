"""Experiment 97 — Can the verification OBJECTIVE improve impact pruning?

Impact pruning drops experts by individual contribution sum_e ||g_e E_e||, which is the
loose triangle bound of the quantity verification actually certifies: the LAYER
deviation ||sum_{e in S} g_e E_e||. Triangle ignores CANCELLATION between dropped
experts. So we test CANCELLATION-AWARE (joint) pruning: greedily build the dropped set
to minimize the true joint layer deviation ||sum_{e in dropped} g_e E_e|| (mean over
tokens), exploiting experts whose gated outputs partially cancel. Compare to
IMPACT-greedy (ascending individual contribution) at matched count, held-out task
accuracy (code, prose).
  CANCEL > IMPACT => the certified layer-deviation objective yields a better selection
  than impact (a real verification-derived improvement, not just a certificate).

GPU, live OLMoE. Run: python3 experiments/97_cancellation_aware_pruning.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 256
N_EVAL = 512
CHUNK = 256
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
def gated_outputs(mlp, H, dev):
    """gE[t,e,:] = g_e(h_t) E_e(h_t) for selected (else 0); [T,N,d]."""
    Wg = mlp.gate.weight.float()
    lg = H @ Wg.T
    K, N = mlp.top_k, mlp.num_experts
    topv, topi = lg.topk(K, dim=-1); g = torch.softmax(topv, dim=-1)
    T, d = H.shape
    gE = torch.zeros(T, N, d, device=dev)
    for e in range(N):
        sel = (topi == e)
        if not sel.any():
            continue
        ti = sel.any(1).nonzero().flatten()
        ge = g[sel]                                   # gate where selected
        W1 = mlp.experts[e].gate_proj.weight.float()
        W2 = mlp.experts[e].up_proj.weight.float()
        W3 = mlp.experts[e].down_proj.weight.float()
        Ee = (F.silu(H[ti] @ W1.T) * (H[ti] @ W2.T)) @ W3.T
        gE[ti, e] = ge.unsqueeze(1) * Ee
    return gE


def impact_order(gE):
    return torch.argsort(gE.norm(dim=2).mean(0)).tolist()   # ascending mean ||g_e E_e||


def cancel_order(gE, maxk):
    """greedy: add expert minimizing mean_t ||sum_dropped + gE[:,e]||."""
    T, N, d = gE.shape
    dropped, mask = [], torch.zeros(N, dtype=torch.bool, device=gE.device)
    dsum = torch.zeros(T, d, device=gE.device)
    for _ in range(maxk):
        cand = dsum.unsqueeze(1) + gE                       # [T,N,d]
        cost = cand.norm(dim=2).mean(0)                     # [N]
        cost[mask] = float("inf")
        e = int(cost.argmin().item())
        dropped.append(e); mask[e] = True
        dsum = dsum + gE[:, e]
    rest = [e for e in range(N) if not mask[e]]
    return dropped + rest


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
    maxk = max(KGRID)

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
        imp_o, can_o = {}, {}
        for li in range(nL):
            gE = gated_outputs(layers[li].mlp, Hc[li], dev)
            imp_o[li] = impact_order(gE)
            can_o[li] = cancel_order(gE, maxk)
            del gE
        torch.cuda.empty_cache() if dev == "cuda" else None

        @torch.no_grad()
        def acc():
            a = []
            for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
                x = ids[c0:c0 + CHUNK].unsqueeze(0)
                p = model(x).logits[0, :-1].float().argmax(-1)
                a.append((p == x[0, 1:]).float().mean().item() * 100)
            return st.mean(a)

        base = acc()
        print(f"\n{'='*60}\n  DOMAIN={name}: held-out top-1 acc (%), baseline={base:.1f}"
              f"\n{'='*60}")
        print(f"  {'%pruned':>7s} | {'IMPACT':>7s} | {'CANCEL-aware':>12s} | {'gain':>6s}")
        print("  " + "-"*44)
        for k in KGRID:
            prune(imp_o, k); ai = acc(); restore()
            prune(can_o, k); ac = acc(); restore()
            print(f"  {k/Nexp*100:>6.0f}% | {ai:>7.1f} | {ac:>12.1f} | {ac-ai:>+6.1f}",
                  flush=True)

    print(f"\n  CANCEL-aware > IMPACT => minimizing the certified layer-deviation objective")
    print("  (with cancellation) is a better pruning selection than individual impact.")


if __name__ == "__main__":
    main()
