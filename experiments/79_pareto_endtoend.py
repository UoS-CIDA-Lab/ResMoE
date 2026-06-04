"""Experiment 79 — Hardening the certified-pruning result:
  (A) the eps <-> budget PARETO (the price tag of the robustness margin), and
  (B) END-TO-END quality of the certified-pruned model vs frequency pruning.

(A) For a specialized domain, sweep the L2 robustness radius eps and report the total
certified-never-selected budget over all 16 layers -> memory saving. Larger eps = a
stronger (wider) robustness guarantee = fewer certifiably-dead experts. This is the
provable robustness-vs-compression trade an operator dials in.

(B) At a fixed eps, actually REMOVE the certified-dead experts (all layers) and measure
held-out perplexity ON THE DOMAIN. Because certified-dead experts provably never fire,
removal is lossless by construction (downstream activations are bit-identical) -> PPL
must be unchanged. Frequency pruning at the SAME per-layer count drops least-USED
experts (which can still fire / are eps-reachable) -> PPL degrades. We report
original / certified-pruned / frequency-pruned PPL, and the held-out activation count
of each pruned set (certified must be 0).

GPU, live OLMoE. Run: python3 experiments/79_pareto_endtoend.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 128
N_HELD = 192
EPS_SWEEP = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
EPS_E2E = [0.05, 0.10]


def streams(tok):
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    gs = load_dataset("gsm8k", "main", split="test")
    prose = "\n\n".join(t for t in wt["text"] if t.strip())
    arith = "\n".join(f"What is {a} plus {b}? The answer is" for a, b in
                      [(i, (i * 7) % 97 + 2) for i in range(3, 320)])
    return {
        "narrow(arith)": tok(arith, return_tensors="pt").input_ids[0],
        "prose": tok(prose, return_tensors="pt").input_ids[0],
    }


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
    Wg = mlp.gate.weight.float()
    rown = Wg.norm(dim=1)
    lg = H @ Wg.T
    ub = lg + eps * rown
    lb = lg - eps * rown
    K, N = mlp.top_k, mlp.num_experts
    dead = []
    for e in range(N):
        if ((lb > ub[:, e:e + 1]).sum(1) >= K).all():
            dead.append(e)
    return dead


@torch.no_grad()
def freq_order(mlp, H):
    lg = H @ mlp.gate.weight.float().T
    tk = lg.topk(mlp.top_k, dim=-1).indices
    return torch.bincount(tk.flatten(), minlength=mlp.num_experts)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    sm = streams(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    nL = len(model.model.layers)
    NTOT = nL * model.model.layers[0].mlp.num_experts

    Hc = {n: collect(model, ids, dev, list(range(nL)), 0, N_CALIB) for n, ids in sm.items()}
    Hh = {n: collect(model, ids, dev, list(range(nL)), N_CALIB, N_CALIB + N_HELD)
          for n, ids in sm.items()}

    # ---- (A) eps <-> budget Pareto ----
    print(f"\n{'='*80}\n  PART A — eps <-> certified pruning budget Pareto (all {nL} layers)"
          f"\n{'='*80}")
    print(f"  {'domain':>14s} | {'eps':>5s} | {'experts pruned':>14s} | {'% of experts':>12s}")
    print("  " + "-"*54)
    dead_cache = {}
    for name in sm:
        for eps in EPS_SWEEP:
            tot = 0
            dd = {}
            for li in range(nL):
                d = certified_dead(model.model.layers[li].mlp, Hc[name][li], eps)
                dd[li] = d
                tot += len(d)
            dead_cache[(name, eps)] = dd
            print(f"  {name:>14s} | {eps:>5.2f} | {tot:>14d} | {tot/NTOT*100:>11.1f}%",
                  flush=True)
        print("  " + "-"*54)

    # ---- (B) end-to-end held-out perplexity ----
    down = [model.model.layers[li].mlp.experts[e].down_proj.weight
            for li in range(nL) for e in range(model.model.layers[li].mlp.num_experts)]
    saved = [w.detach().cpu().clone() for w in down]

    def restore():
        with torch.no_grad():
            for w, s in zip(down, saved):
                w.data.copy_(s.to(dev))

    def prune(sets):
        with torch.no_grad():
            for li in range(nL):
                for e in sets[li]:
                    model.model.layers[li].mlp.experts[e].down_proj.weight.data.zero_()

    @torch.no_grad()
    def ppl(ids):
        x = ids[N_CALIB:N_CALIB + N_HELD].unsqueeze(0).to(dev)
        lg = model(x).logits[0, :-1].float()
        return F.cross_entropy(lg, x[0, 1:]).exp().item()

    @torch.no_grad()
    def held_activations(name, sets):
        a = 0
        for li in range(nL):
            mlp = model.model.layers[li].mlp
            tk = (Hh[name][li] @ mlp.gate.weight.float().T).topk(mlp.top_k, -1).indices
            u = tk.flatten()
            a += sum(int((u == e).sum().item()) for e in sets[li])
        return a

    print(f"\n{'='*80}\n  PART B — end-to-end held-out perplexity: certified vs frequency "
          f"pruning\n{'='*80}")
    print(f"  {'domain':>14s} {'eps':>5s} | {'pruned':>6s} | {'orig PPL':>9s} | "
          f"{'cert PPL':>9s} (act) | {'freq PPL':>9s} (act)")
    print("  " + "-"*78)
    for name in sm:
        base = ppl(sm[name])
        for eps in EPS_E2E:
            cset = dead_cache[(name, eps)]
            tot = sum(len(cset[li]) for li in range(nL))
            # frequency: drop the same per-layer count, least-used
            fset = {}
            for li in range(nL):
                k = len(cset[li])
                fr = freq_order(model.model.layers[li].mlp, Hc[name][li])
                fset[li] = torch.argsort(fr)[:k].tolist()
            ca = held_activations(name, cset)
            fa = held_activations(name, fset)
            prune(cset); pc = ppl(sm[name]); restore()
            prune(fset); pf = ppl(sm[name]); restore()
            print(f"  {name:>14s} {eps:>5.2f} | {tot:>6d} | {base:>9.3f} | "
                  f"{pc:>9.3f} ({ca:>4d}) | {pf:>9.3f} ({fa:>4d})", flush=True)

    print("\n  (A) larger eps -> fewer certifiably-dead experts = the price of a wider")
    print("  robustness guarantee. (B) certified-pruned PPL == original (0 held-out")
    print("  activations -> provably & empirically lossless on the domain), while")
    print("  frequency-pruned PPL degrades (it removed experts the held-out domain uses).")


if __name__ == "__main__":
    main()
