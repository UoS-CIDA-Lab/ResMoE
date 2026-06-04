"""Experiment 90 — Does CERTIFIED pruning give measurably BETTER output quality than
unguaranteed (frequency) pruning, on the inputs where they differ?

certified pruning removes only experts PROVABLY never selected over the operational
eps-region; frequency pruning removes the least-USED experts on calibration, some of
which are eps-REACHABLE (Z\\C, exp 78) -> they DO fire on some held-out inputs. On the
held-out positions where such an expert fires, frequency-pruned (which dropped it)
must deviate from the original, while certified-pruned (which kept it) matches the
original. We measure model output quality (NLL vs the original model's prediction) at
THOSE positions for: original, certified-pruned, frequency-pruned. If
frequency-pruned NLL > certified-pruned (~original) there, certified is measurably
better -> the minimal demonstration the user asks for. If equal, MoE redundancy absorbs
even frequency's unsafe drops -> certified is equal-quality-with-a-proof, not better.

Narrow domain (arithmetic template), matched per-layer pruning count. GPU, live OLMoE.
Run: python3 experiments/90_certified_quality_advantage.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 128
N_EVAL = 192
EPS = 0.05


def stream(tok):
    arith = "\n".join(f"What is {a} plus {b}? The answer is" for a, b in
                      [(i, (i * 7) % 97 + 2) for i in range(3, 400)])
    return tok(arith, return_tensors="pt").input_ids[0]


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
def certified_dead(mlp, H, eps):
    Wg = mlp.gate.weight.float(); rown = Wg.norm(dim=1)
    lg = H @ Wg.T
    ub = lg + eps * rown; lb = lg - eps * rown
    K, N = mlp.top_k, mlp.num_experts
    return [e for e in range(N) if ((lb > ub[:, e:e+1]).sum(1) >= K).all()]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = stream(tok).to(dev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers
    nL = len(layers)

    Hc = collect(model, ids, dev, list(range(nL)), 0, N_CALIB)
    Hh = collect(model, ids, dev, list(range(nL)), N_CALIB, N_CALIB + N_EVAL)

    cert_set, freq_set = {}, {}
    # positions (layer, token) where a freq-dropped-but-cert-kept expert fires on held-out
    risky_positions = []     # (layer, tokenpos)
    for li in range(nL):
        mlp = layers[li].mlp
        C = certified_dead(mlp, Hc[li], EPS)
        cert_set[li] = set(C)
        k = len(C)
        lg = Hc[li] @ mlp.gate.weight.float().T
        tk = lg.topk(mlp.top_k, -1).indices
        freq = torch.bincount(tk.flatten(), minlength=mlp.num_experts)
        F_ = torch.argsort(freq)[:k].tolist()
        freq_set[li] = set(F_)
        risky = set(F_) - set(C)            # freq drops these, cert keeps them
        if risky:
            th = Hh[li] @ mlp.gate.weight.float().T
            thi = th.topk(mlp.top_k, -1).indices       # [T_eval, K]
            for e in risky:
                fires = (thi == e).any(-1).nonzero().flatten().tolist()
                for t in fires:
                    risky_positions.append((li, t))
    tot_cert = sum(len(c) for c in cert_set.values())
    print(f"  pruned per model: {tot_cert} experts (matched). risky firings on held-out "
          f"(freq-dropped, cert-kept, actually active): {len(risky_positions)}", flush=True)

    # token positions (in held-out) touched by a risky firing
    risky_tokens = sorted(set(t for _, t in risky_positions))

    down = [layers[li].mlp.experts[e].down_proj.weight
            for li in range(nL) for e in range(layers[li].mlp.num_experts)]
    saved = [w.detach().cpu().clone() for w in down]

    def restore():
        with torch.no_grad():
            for w, s in zip(down, saved):
                w.data.copy_(s.to(dev))

    def prune(sets):
        with torch.no_grad():
            for li in range(nL):
                for e in sets[li]:
                    layers[li].mlp.experts[e].down_proj.weight.data.zero_()

    @torch.no_grad()
    def logprobs():
        x = ids[N_CALIB:N_CALIB + N_EVAL].unsqueeze(0).to(dev)
        lg = model(x).logits[0, :-1].float().log_softmax(-1)   # [pos, V]
        tgt = x[0, 1:]
        return lg[torch.arange(len(tgt)), tgt]                 # logprob of true next tok

    base_lp = logprobs()
    prune(cert_set); cert_lp = logprobs(); restore()
    prune(freq_set); freq_lp = logprobs(); restore()

    def nll(lp, idx=None):
        v = -lp if idx is None else -lp[idx]
        return v.mean().item()

    ridx = torch.tensor([t for t in risky_tokens if t < len(base_lp)], device=dev)
    print(f"\n{'='*70}\n  Output quality (mean NLL vs true next token; lower=better)"
          f"\n{'='*70}")
    print(f"  {'set':>26s} | {'ALL held-out':>13s} | {'risky positions':>16s}")
    print("  " + "-"*60)
    print(f"  {'original (fp16)':>26s} | {nll(base_lp):>13.4f} | "
          f"{nll(base_lp, ridx) if len(ridx) else float('nan'):>16.4f}")
    print(f"  {'certified-pruned':>26s} | {nll(cert_lp):>13.4f} | "
          f"{nll(cert_lp, ridx) if len(ridx) else float('nan'):>16.4f}")
    print(f"  {'frequency-pruned':>26s} | {nll(freq_lp):>13.4f} | "
          f"{nll(freq_lp, ridx) if len(ridx) else float('nan'):>16.4f}", flush=True)

    if len(ridx):
        adv = nll(freq_lp, ridx) - nll(cert_lp, ridx)
        print(f"\n  freq - cert NLL on risky positions = {adv:+.4f}")
        print(f"  > 0  => certified is MEASURABLY BETTER where they differ (the demonstration).")
        print(f"  ~ 0  => redundancy absorbs freq's unsafe drops -> equal quality, only the")
        print(f"          proof differs (certified is equal-quality-with-a-guarantee).")
    else:
        print("\n  no risky firings -> certified and frequency pruned identical sets here.")


if __name__ == "__main__":
    main()
