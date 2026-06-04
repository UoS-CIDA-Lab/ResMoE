"""Experiment 91 — Does the small NLL gap (exp 90) become a meaningful DISCRETE
task-performance gap? certified vs frequency pruning.

exp 90: on positions where a frequency-dropped-but-certified-kept expert fires
('risky'), certified NLL < frequency NLL by ~0.02. NLL is continuous; task accuracy is
discrete (a prediction flips only if the gap crosses a decision boundary). We measure
DISCRETE metrics on a narrow arithmetic domain (the task = predict the correct next
token / answer):
  - top-1 next-token ACCURACY (pred == true) for original / certified / frequency,
    overall AND on risky positions;
  - agreement-with-ORIGINAL top-1 (which pruned model behaves like the reference);
  - certified-vs-frequency top-1 DISAGREEMENT on risky positions (behavioral divergence).
We also sweep pruning aggressiveness (eps controls the certified budget; we compare at
matched per-layer count) to see whether the gap widens as redundancy is exhausted.

GPU, live OLMoE. Run: python3 experiments/91_task_level_advantage.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

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

    cert_set, freq_set, risky_tokens = {}, {}, set()
    for li in range(nL):
        mlp = layers[li].mlp
        C = certified_dead(mlp, Hc[li], EPS)
        cert_set[li] = set(C)
        k = len(C)
        tk = (Hc[li] @ mlp.gate.weight.float().T).topk(mlp.top_k, -1).indices
        freq = torch.bincount(tk.flatten(), minlength=mlp.num_experts)
        F_ = torch.argsort(freq)[:k].tolist()
        freq_set[li] = set(F_)
        risky = set(F_) - set(C)
        if risky:
            thi = (Hh[li] @ mlp.gate.weight.float().T).topk(mlp.top_k, -1).indices
            for e in risky:
                for t in (thi == e).any(-1).nonzero().flatten().tolist():
                    risky_tokens.add(t)
    tot = sum(len(c) for c in cert_set.values())

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
    def preds():
        x = ids[N_CALIB:N_CALIB + N_EVAL].unsqueeze(0).to(dev)
        lg = model(x).logits[0, :-1].float()
        return lg.argmax(-1), x[0, 1:]                  # pred top-1, true next

    base_p, tgt = preds()
    prune(cert_set); cert_p, _ = preds(); restore()
    prune(freq_set); freq_p, _ = preds(); restore()

    ridx = torch.tensor(sorted(t for t in risky_tokens if t < len(tgt)), device=dev)

    def acc(p, idx=None):
        m = (p == tgt) if idx is None else (p[idx] == tgt[idx])
        return m.float().mean().item() * 100

    def agree(p, q, idx=None):
        m = (p == q) if idx is None else (p[idx] == q[idx])
        return m.float().mean().item() * 100

    print(f"  pruned {tot} experts (matched); risky token positions on held-out: "
          f"{len(ridx)}/{len(tgt)}", flush=True)
    print(f"\n{'='*72}\n  DISCRETE next-token top-1 metrics (%)\n{'='*72}")
    print(f"  {'metric':>34s} | {'ALL':>7s} | {'risky pos':>9s}")
    print("  " + "-"*58)
    print(f"  {'accuracy: original':>34s} | {acc(base_p):>7.1f} | {acc(base_p, ridx):>9.1f}")
    print(f"  {'accuracy: certified-pruned':>34s} | {acc(cert_p):>7.1f} | "
          f"{acc(cert_p, ridx):>9.1f}")
    print(f"  {'accuracy: frequency-pruned':>34s} | {acc(freq_p):>7.1f} | "
          f"{acc(freq_p, ridx):>9.1f}")
    print(f"  {'agree-with-original: certified':>34s} | {agree(cert_p, base_p):>7.1f} | "
          f"{agree(cert_p, base_p, ridx):>9.1f}")
    print(f"  {'agree-with-original: frequency':>34s} | {agree(freq_p, base_p):>7.1f} | "
          f"{agree(freq_p, base_p, ridx):>9.1f}")
    print(f"  {'certified-vs-frequency disagree':>34s} | "
          f"{100-agree(cert_p, freq_p):>7.1f} | {100-agree(cert_p, freq_p, ridx):>9.1f}",
          flush=True)
    print(f"\n  If certified accuracy/agreement > frequency on risky positions, the small")
    print("  NLL gap (exp 90) DOES translate to a discrete behavioral/task advantage there.")
    print("  certified-vs-frequency disagreement quantifies how often they actually differ.")


if __name__ == "__main__":
    main()
