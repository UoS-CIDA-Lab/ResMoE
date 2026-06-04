"""Experiment 62 — Task-conditioned certified budget + certified routing stability.

Knowing the task lets us compress more, via two MoE-specialization effects:
  (A) SUPPORT REDUCTION: experts never in topK on task tokens contribute 0 to
      every per-token constraint -> free to push to min bits / prune. |A_T| << N.
  (B) GATE SHARPENING: within A_T the gate mass concentrates -> the router-aware
      greedy spends precision on fewer experts at the same delta_lyr.
The certificate's SCOPE narrows to the task input region R_T (a conditional
certificate, sound for task-specialized deployment).

SOUNDNESS of support reduction requires CERTIFIED ROUTING STABILITY: an
eps-perturbation must not pull an out-of-A_T expert into topK. The router is
linear (h -> W_g h), so over B2(h,eps) each logit moves by <= eps*||w_e||. The
topK SET is certified-invariant at token t iff
    min_{e in S}(logit_e - eps||w_e||) > max_{e not in S}(logit_e + eps||w_e||).
We report the fraction of task tokens with certified-stable routing.

Compares tasks {code, math, prose} vs generic. GPU, 75% cap.
RUN AFTER exp 60 frees the GPU. python3 experiments/62_task_conditioned.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
EPS = 0.05
DELTA_LYR = 0.01
BITS = [16, 8, 6, 4, 3, 2]
N_CALIB = 512
STAT_LAYERS = [0, 7, 15]      # layers for support/stability stats
BUDGET_LAYERS = [0, 7, 15]    # detailed task-conditioned budget on these layers


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def silu_grad(z):
    s = torch.sigmoid(z)
    return s + z * s * (1 - s)


def affine_box(W, lo, hi):
    c = (lo + hi) / 2; r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_consts(dev):
    z = torch.linspace(-30, 30, 400001, device=dev).requires_grad_(True)
    g1, = torch.autograd.grad(F.silu(z).sum(), z, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    return (g1.abs().max().item()*1.01, g2.abs().max().item()*1.01,
            g3.abs().max().item()*1.01)


def quant(W, bits):
    if bits >= 16:
        return W
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def spec(W):
    return torch.linalg.matrix_norm(W.float(), ord=2).item()


@torch.no_grad()
def expert_bound(W1, W2, W3, W1q, W2q, W3q, h0, eps, sv, Ls, Lpp, Lppp):
    s1, s2, s1q, s2q, sd1, sd2 = sv
    d1, d2, d3 = W1 - W1q, W2 - W2q, W3 - W3q
    lo, hi = h0 - eps, h0 + eps
    amax = lambda a, b: torch.maximum(a.abs(), b.abs())
    Bv = amax(*affine_box(W2, lo, hi)); Bvq = amax(*affine_box(W2q, lo, hi))
    Bd1 = amax(*affine_box(d1, lo, hi)); Bd2 = amax(*affine_box(d2, lo, hi))
    Dc = Lpp * (W3.abs() * Bv.unsqueeze(0)).amax(1)
    De = Ls * W3.abs().amax(1)
    dDc = (W3.abs()*(Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0)
           + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs()*(Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    HD = Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)
    def J(A, B, C):
        u = A @ h0; v = B @ h0
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JD = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0.unsqueeze(0), W1, W2, W3)
           - swiglu(h0.unsqueeze(0), W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps*JD + 0.5*eps*eps*HD).max().item()


@torch.no_grad()
def routing_stats(block, Hc, dev, eps):
    """Return (|A_T|, frac certified-stable routing, mean top-K gate mass)."""
    K, N = block.top_k, block.num_experts
    Wg = block.gate.weight.float()                  # [N, d]
    rownorm = Wg.norm(dim=1)                         # [N]
    logits = Hc @ Wg.T                               # [T, N]
    topv, topk = logits.topk(K, dim=-1)
    gates = torch.softmax(topv, dim=-1)
    T = Hc.shape[0]
    inset = torch.zeros(T, N, dtype=torch.bool, device=dev)
    inset.scatter_(1, topk, True)
    A = int((inset.any(0)).sum().item())
    # certified set-invariance over B2(h,eps): linear-router logit interval
    lo = logits - eps * rownorm                      # [T,N]
    hi = logits + eps * rownorm
    sel_lo = lo.masked_fill(~inset, float("inf")).amin(1)   # min selected lower
    uns_hi = hi.masked_fill(inset, float("-inf")).amax(1)   # max unselected upper
    stable = (sel_lo > uns_hi)
    frac = stable.float().mean().item()
    gate_mass = gates.sum(1).mean().item()
    return A, frac, gate_mass


@torch.no_grad()
def task_budget(block, Hc, dev, Ls, Lpp, Lppp):
    """Task-conditioned router-aware budget; returns (avg over N, avg over A_T, |A_T|)."""
    K, N = block.top_k, block.num_experts
    Wg = block.gate.weight.float()
    logits = Hc @ Wg.T
    topv, topk = logits.topk(K, dim=-1)
    gates = torch.softmax(topv, dim=-1)
    T = Hc.shape[0]
    inset = torch.zeros(T, N, dtype=torch.bool, device=dev); inset.scatter_(1, topk, True)
    active = [e for e in range(N) if inset[:, e].any()]
    # NO cap on routed inputs: D_e must bound ALL inputs the expert serves (sound).
    routed = {e: torch.nonzero(inset[:, e]).flatten().tolist() for e in active}
    cert = {e: {16: 0.0} for e in active}
    for e in active:
        W1 = block.experts[e].gate_proj.weight.float()
        W2 = block.experts[e].up_proj.weight.float()
        W3 = block.experts[e].down_proj.weight.float()
        s1, s2 = spec(W1), spec(W2)
        for b in BITS:
            if b >= 16:
                continue
            W1q, W2q, W3q = quant(W1, b), quant(W2, b), quant(W3, b)
            sv = (s1, s2, spec(W1q), spec(W2q), spec(W1-W1q), spec(W2-W2q))
            cert[e][b] = max(expert_bound(W1, W2, W3, W1q, W2q, W3q, Hc[t], EPS, sv,
                                          Ls, Lpp, Lppp) for t in routed[e])
    bits = {e: 2 for e in active}
    nxt = {2: 3, 3: 4, 4: 6, 6: 8, 8: 16, 16: 16}
    for _ in range(20000):
        worst_t, worst_s = -1, DELTA_LYR
        for t in range(T):
            s = sum(gates[t, j].item()*cert[topk[t, j].item()][bits[topk[t, j].item()]]
                    for j in range(K))
            if s > worst_s:
                worst_s, worst_t = s, t
        if worst_t < 0:
            break
        cand = [(gates[worst_t, j].item()*cert[topk[worst_t, j].item()][bits[topk[worst_t, j].item()]],
                 topk[worst_t, j].item()) for j in range(K)
                if bits[topk[worst_t, j].item()] != 16]
        if not cand:
            break
        e = max(cand)[1]
        bits[e] = nxt[bits[e]]
    sum_active = sum(bits.values())
    avg_A = sum_active / len(active)
    # inactive experts -> 2-bit (free under the task certificate)
    avg_N = (sum_active + 2 * (N - len(active))) / N
    return avg_N, avg_A, len(active)


@torch.no_grad()
def collect_calib(model, ids, dev, layers, n):
    caps = {li: [] for li in layers}
    hs = []
    for li in layers:
        def mk(li):
            def hk(_m, a):
                caps[li].append(a[0].detach().reshape(-1, a[0].shape[-1]).float())
            return hk
        hs.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    try:
        model(ids[:n].unsqueeze(0).to(dev))
    finally:
        for h in hs:
            h.remove()
    return {li: torch.cat(caps[li]) for li in layers}


def task_streams(tok):
    from datasets import load_dataset
    streams = {}
    try:
        wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        streams["prose"] = "\n\n".join(t for t in wt["text"] if t.strip())
    except Exception as e:
        print(f"  prose load failed: {e}")
    try:
        he = load_dataset("openai_humaneval", split="test")
        streams["code"] = "\n\n".join(he["prompt"])
    except Exception as e:
        print(f"  code load failed: {e}")
    try:
        gs = load_dataset("gsm8k", "main", split="test")
        streams["math"] = "\n\n".join(gs["question"][:400])
    except Exception as e:
        print(f"  math load failed: {e}")
    return {k: tok(v, return_tensors="pt").input_ids[0] for k, v in streams.items()}


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    streams = task_streams(tok)
    print(f"Tasks: {list(streams)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    Ls, Lpp, Lppp = silu_consts(dev)

    print(f"\n{'='*78}\n  Routing stats per task (eps_L2={EPS}); N=64 experts, top-8\n{'='*78}")
    print(f"  {'task':7s} {'layer':>5s} {'|A_T|':>6s} {'cert-stable routing':>20s} {'gate mass':>10s}")
    budget_rows = []
    for task, ids in streams.items():
        calib = collect_calib(model, ids, dev, STAT_LAYERS, N_CALIB)
        for li in STAT_LAYERS:
            A, frac, gm = routing_stats(model.model.layers[li].mlp, calib[li], dev, EPS)
            print(f"  {task:7s} {li:>5d} {A:>6d} {frac*100:>18.1f}% {gm:>10.3f}", flush=True)
        # detailed budget on each BUDGET_LAYERS layer
        for li in BUDGET_LAYERS:
            avg_N, avg_A, nA = task_budget(model.model.layers[li].mlp,
                                           calib[li], dev, Ls, Lpp, Lppp)
            budget_rows.append((task, li, nA, avg_A, avg_N))
            print(f"  -> {task} layer{li} budget computed", flush=True)

    print(f"\n{'='*78}\n  Task-conditioned router-aware budget (delta={DELTA_LYR})"
          f"\n{'='*78}")
    print(f"  {'task':7s} {'layer':>5s} {'|A_T|':>6s} {'bits/active':>12s} "
          f"{'bits/all-64':>12s} {'mem (all)':>10s}")
    for task, li, nA, avg_A, avg_N in budget_rows:
        print(f"  {task:7s} {li:>5d} {nA:>6d} {avg_A:>10.2f}b {avg_N:>10.2f}b "
              f"{avg_N/16*100:>9.0f}%")
    print(f"\n  bits/all-64 counts out-of-support experts at 2-bit (free under the")
    print("  task certificate); the gap vs generic is the support-reduction payoff.")
    print("  cert-stable routing% is the fraction of tokens whose top-K set is")
    print("  PROVABLY invariant over B2(h,eps) -> support reduction is sound there.")


if __name__ == "__main__":
    main()
