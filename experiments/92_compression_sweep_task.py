"""Experiment 92 — Does a HIGHER compression rate make the certified selection
criterion beat the frequency criterion on task performance (as redundancy is
exhausted)? User's hypothesis.

At mild compression (exp 91) certified and frequency pruning give identical discrete
task output (redundancy absorbs frequency's unsafe drops). Maybe at AGGRESSIVE
compression redundancy is exhausted and the safety-based selection wins. We sweep the
per-layer drop count k and compare held-out arithmetic next-token top-1 ACCURACY for:
  - original,
  - CERT-order pruning: drop provably-never-selected experts first (exactly lossless),
    then by ascending certified contribution (safest next),
  - FREQ-order pruning: drop the least-FREQUENT experts first (the unguaranteed
    heuristic).
Both drop the same count k. If CERT-order accuracy stays above FREQ-order as k grows,
the certified criterion is a better selection under aggressive compression. Note the
certificate's GUARANTEE only holds up to the certified-dead budget; beyond it CERT-order
is just a (still safety-ordered) heuristic with no proof.

GPU, live OLMoE. Run: python3 experiments/92_compression_sweep_task.py
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
KGRID = [16, 24, 32, 40, 48, 56]


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
def cert_and_contrib(mlp, H, eps, dev):
    """return (certified-dead set, per-expert contribution g*||E|| for ordering)."""
    Wg = mlp.gate.weight.float(); rown = Wg.norm(dim=1)
    lg = H @ Wg.T
    ub = lg + eps * rown; lb = lg - eps * rown
    K, N = mlp.top_k, mlp.num_experts
    dead = [e for e in range(N) if ((lb > ub[:, e:e+1]).sum(1) >= K).all()]
    topv, topi = lg.topk(K, dim=-1); g = torch.softmax(topv, dim=-1)
    contrib = torch.full((N,), 1e9, device=dev)
    for e in range(N):
        sel = (topi == e)
        if sel.any():
            ti = sel.any(1).nonzero().flatten()
            W1 = mlp.experts[e].gate_proj.weight.float()
            W2 = mlp.experts[e].up_proj.weight.float()
            W3 = mlp.experts[e].down_proj.weight.float()
            Ee = (F.silu(H[ti] @ W1.T) * (H[ti] @ W2.T)) @ W3.T
            contrib[e] = (g[sel] * Ee.norm(dim=-1)).mean()
        else:
            contrib[e] = 0.0
    return set(dead), contrib


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
    Nexp = layers[0].mlp.num_experts

    Hc = collect(model, ids, dev, list(range(nL)), 0, N_CALIB)
    cert_order, freq_order, cert_budget = {}, {}, []
    for li in range(nL):
        mlp = layers[li].mlp
        dead, contrib = cert_and_contrib(mlp, Hc[li], EPS, dev)
        cert_budget.append(len(dead))
        # cert-order: dead first (key 0), then ascending contribution
        key = [(0 if e in dead else 1, contrib[e].item()) for e in range(Nexp)]
        cert_order[li] = sorted(range(Nexp), key=lambda e: key[e])
        tk = (Hc[li] @ mlp.gate.weight.float().T).topk(mlp.top_k, -1).indices
        freq = torch.bincount(tk.flatten(), minlength=Nexp)
        freq_order[li] = torch.argsort(freq).tolist()
    avg_budget = sum(cert_budget) / nL
    print(f"  avg certified-dead budget/layer = {avg_budget:.1f}/{Nexp}", flush=True)

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
    def acc():
        x = ids[N_CALIB:N_CALIB + N_EVAL].unsqueeze(0).to(dev)
        p = model(x).logits[0, :-1].float().argmax(-1)
        return (p == x[0, 1:]).float().mean().item() * 100

    base = acc()
    print(f"\n{'='*64}\n  Held-out task (next-token top-1 acc, %) vs compression"
          f"\n  baseline={base:.1f}; certified GUARANTEE holds for k <= {avg_budget:.0f}"
          f"\n{'='*64}")
    print(f"  {'k/layer':>7s} | {'% experts pruned':>16s} | {'CERT-order':>10s} | "
          f"{'FREQ-order':>10s} | {'cert-freq':>9s}")
    print("  " + "-"*62)
    for k in KGRID:
        prune(cert_order, k); ac = acc(); restore()
        prune(freq_order, k); af = acc(); restore()
        flag = "  (within budget)" if k <= avg_budget else "  (>budget: uncertified)"
        print(f"  {k:>7d} | {k/Nexp*100:>15.0f}% | {ac:>10.1f} | {af:>10.1f} | "
              f"{ac-af:>+8.1f}{flag}", flush=True)

    print(f"\n  If CERT-order > FREQ-order grows with k, the safety-based selection wins")
    print("  under aggressive compression. If ~0, redundancy absorbs both even when")
    print("  exhausted, and certified's value remains the proof (within budget), not accuracy.")


if __name__ == "__main__":
    main()
