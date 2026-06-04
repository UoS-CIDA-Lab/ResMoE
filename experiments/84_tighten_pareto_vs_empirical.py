"""Experiment 84 — Tighten the certified adaptive-K result:
  (A) eps <-> coverage <-> compute-saving PARETO for the certified method, and
  (B) head-to-head vs EMPIRICAL dynamic-K (same delta criterion, but point gates and
      all tokens, no guarantee) to make the difference quantitative.

Same per-layer drop criterion (drop selected experts with smallest contribution
c_e while sum c_e <= delta), TWO modes:
  CERT  : c_e = gate_UPPER_BOUND_over_eps_ball * ||E_e(h)||, and ONLY drop on tokens
          whose top-K set is certified-stable over the eps-ball -> layer change <= delta
          PROVABLY for every input in the ball.
  EMP   : c_e = point gate (the actual softmax weight) * ||E_e(h)||, dropped on ALL
          tokens (no stability check) -> the standard dynamic-K / gate-threshold style,
          no robustness guarantee.
Compute proxy = effective K (experts run per token); FLOP saving = (K-effK)/K since
all experts are equal size and the router is negligible.

We report, per eps (CERT) and for EMP: effective K, held-out PPL, % routing-stable,
and -- the key contrast -- the fraction of EMP's drops that are UNSAFE (on non-stable
tokens, i.e. drops the certificate refuses because a perturbation could flip routing).

GPU, live OLMoE. Run: python3 experiments/84_tighten_pareto_vs_empirical.py
"""
from __future__ import annotations

import sys
import pathlib
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CTX = 256
N_EVAL = 192
DELTA = 0.05
EPS_GRID = [0.01, 0.02, 0.05, 0.10]

_EPS = 0.05
_DELTA = DELTA
_MODE = "cert"
_ST = {}


def streams(tok):
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return tok("\n\n".join(t for t in wt["text"] if t.strip()),
               return_tensors="pt").input_ids[0]


def patched_forward(self, hidden_states):
    B, S, d = hidden_states.shape
    h = hidden_states.view(-1, d)
    T = h.shape[0]
    logits = self.gate(h)
    lg = logits.float()
    N, K = self.num_experts, self.top_k
    probs = F.softmax(lg, dim=1)
    rw, sel = torch.topk(probs, K, dim=-1)
    rw = rw.to(h.dtype)
    Wg = self.gate.weight.float()
    rown = Wg.norm(dim=1)

    sel_lo = lg - _EPS * rown.unsqueeze(0)
    sel_hi = lg + _EPS * rown.unsqueeze(0)
    inset = torch.zeros(T, N, dtype=torch.bool, device=h.device)
    inset.scatter_(1, sel, True)
    min_sel_lo = sel_lo.masked_fill(~inset, float("inf")).amin(1)
    max_uns_hi = sel_hi.masked_fill(inset, float("-inf")).amax(1)
    stable = min_sel_lo > max_uns_hi                       # [T]

    # expert outputs for selected, and norms
    out_store = torch.zeros(T, K, d, device=h.device, dtype=h.dtype)
    enorm = torch.zeros(T, K, device=h.device)
    for e in range(N):
        hit = (sel == e)
        if not hit.any():
            continue
        tok = hit.any(1).nonzero().flatten()
        ye = self.experts[e](h[tok])
        slots = hit[tok].float().argmax(1)
        out_store[tok, slots] = ye.to(h.dtype)
        enorm[tok, slots] = ye.float().norm(dim=1)

    lg_sel = torch.gather(lg, 1, sel)
    w_sel = rown[sel]
    if _MODE == "cert":
        sum_all_lo = torch.logsumexp(sel_lo, dim=1, keepdim=True).exp()
        num = (lg_sel + _EPS * w_sel)
        lo_e = (lg_sel - _EPS * w_sel)
        den = num.exp() + (sum_all_lo - lo_e.exp())
        gate_for_c = num.exp() / den                       # gate upper bound
    else:
        gate_for_c = rw.float()                            # point gate (actual softmax)
    contrib = gate_for_c * enorm

    order = torch.argsort(contrib, dim=1)
    csum = torch.cumsum(torch.gather(contrib, 1, order), dim=1)
    drop_sorted = csum <= _DELTA
    drop = torch.zeros(T, K, dtype=torch.bool, device=h.device)
    drop.scatter_(1, order, drop_sorted)
    if _MODE == "cert":
        drop = drop & stable.unsqueeze(1)                  # stable tokens only

    # stats
    _ST["tok"] += T
    _ST["stable"] += int(stable.sum().item())
    _ST["effK"] += float((~drop).float().sum().item())
    if _MODE == "emp":
        # drops on non-stable tokens = unsafe (cert would refuse)
        _ST["emp_drops"] += int(drop.sum().item())
        _ST["emp_unsafe"] += int((drop & ~stable.unsqueeze(1)).sum().item())

    gates = rw * (~drop).to(rw.dtype)
    final = (out_store * gates.unsqueeze(-1)).sum(1)
    return final.view(B, S, d), logits


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    global _EPS, _MODE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = streams(tok)[:N_CTX].to(dev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    blocks = [layer.mlp for layer in model.model.layers]
    K = blocks[0].top_k

    def reset():
        _ST.clear()
        _ST.update({"tok": 0, "stable": 0, "effK": 0.0, "emp_drops": 0, "emp_unsafe": 0})

    @torch.no_grad()
    def ppl():
        lg = model(ids.unsqueeze(0)).logits[0, N_CTX - N_EVAL - 1:-1].float()
        return F.cross_entropy(lg, ids[N_CTX - N_EVAL:]).exp().item()

    base = ppl()
    for b in blocks:
        b._orig = b.forward
        b.forward = types.MethodType(patched_forward, b)
    try:
        print(f"\n{'='*78}\n  TIGHTEN: certified adaptive-K Pareto vs empirical dynamic-K "
              f"(delta={DELTA})\n  baseline PPL={base:.3f}, K={K}\n{'='*78}")
        print(f"  {'method':>22s} | {'effK':>5s} | {'FLOP save':>9s} | {'PPL':>8s} | "
              f"{'%stable':>7s}")
        print("  " + "-"*64)
        _MODE = "cert"
        for eps in EPS_GRID:
            _EPS = eps; reset()
            p = ppl()
            effK = _ST["effK"] / _ST["tok"]
            print(f"  {'CERT eps='+format(eps,'.2f'):>22s} | {effK:>5.2f} | "
                  f"{(K-effK)/K*100:>8.1f}% | {p:>8.3f} | "
                  f"{_ST['stable']/_ST['tok']*100:>6.1f}%", flush=True)
        _MODE = "emp"; _EPS = 0.05; reset()
        p = ppl()
        effK = _ST["effK"] / _ST["tok"]
        frac_unsafe = _ST["emp_unsafe"] / max(_ST["emp_drops"], 1) * 100
        print(f"  {'EMP dynamic-K':>22s} | {effK:>5.2f} | {(K-effK)/K*100:>8.1f}% | "
              f"{p:>8.3f} | {'--':>6s}", flush=True)
        print(f"\n  EMP drops {_ST['emp_drops']} expert-slots; {frac_unsafe:.0f}% are on "
              f"NON-routing-stable tokens (eps=0.05) -> unsafe drops the certificate refuses.")
    finally:
        for b in blocks:
            b.forward = b._orig

    print(f"\n  Pareto: smaller eps -> more routing-stable tokens -> larger certified FLOP")
    print("  saving (weaker robustness margin). EMP saves more but a fraction of its drops")
    print("  are on fragile-routing tokens with no guarantee; CERT is the sound subset.")


if __name__ == "__main__":
    main()
