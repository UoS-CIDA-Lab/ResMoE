"""Experiment 99 — HYBRID certificate: distributional Clopper-Pearson over epsilon-BALLS
certified by the per-component worst-case bound.

The plain distributional certificate (exp 98) only covers sampled POINTS. The user's
idea: make each CP trial an epsilon-BALL and certify it with the tight per-expert
worst-case bound. For quantization the router is shared, so at any input comp and orig
pick the same top-K; over an eps-ball the layer deviation is <= max over REACHABLE
experts of the per-expert eps-ball bound (exp 53/54). A token's ball is "all-layer
certified at delta" if every layer's max-reachable bound <= delta. Clopper-Pearson then
bounds the fraction of the deployment distribution whose eps-ball is NOT certified.
Result: a distributional guarantee over continuous NEIGHBORHOODS (robust to eps
perturbation/shift), not just sampled points -- this is where the verification engines
earn their place inside the distributional framing.

GPU, live OLMoE. Run: python3 experiments/99_hybrid_certificate.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_TOK = 96
EPS = 0.05
CONF = 0.95
BITS = [8, 4]
DELTAS = [0.02, 0.05, 0.1, 0.2]


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


def silu_grad(z):
    s = torch.sigmoid(z); return s + z * s * (1 - s)


def affine_box(W, lo, hi):
    c = (lo + hi) / 2; r = (hi - lo) / 2
    return W @ c - W.abs() @ r, W @ c + W.abs() @ r


def silu_consts(dev):
    z = torch.linspace(-30, 30, 400001, device=dev).requires_grad_(True)
    g1, = torch.autograd.grad(F.silu(z).sum(), z, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    return (g1.abs().max().item()*1.01, g2.abs().max().item()*1.01, g3.abs().max().item()*1.01)


def quant(W, b):
    qmax = 2 ** (b - 1) - 1
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
    dDc = (W3.abs()*(Lpp*Bd2 + Lppp*Bd1*Bvq).unsqueeze(0) + d3.abs()*(Lpp*Bvq).unsqueeze(0)).amax(1)
    dDe = (W3.abs()*(Lpp*Bd1).unsqueeze(0) + d3.abs()*Ls).amax(1)
    HD = Dc*sd1*(s1+s1q) + s1q*s1q*dDc + 2*(s1*De*sd2 + sd1*De*s2q + s1q*dDe*s2q)
    def J(A, B, C):
        u = A @ h0; v = B @ h0
        return (C*(silu_grad(u)*v).unsqueeze(0))@A + (C*F.silu(u).unsqueeze(0))@B
    JD = (J(W1, W2, W3) - J(W1q, W2q, W3q)).norm(dim=1)
    Dh0 = (swiglu(h0.unsqueeze(0), W1, W2, W3) - swiglu(h0.unsqueeze(0), W1q, W2q, W3q)).squeeze(0).abs()
    return (Dh0 + eps*JD + 0.5*eps*eps*HD).max().item()


def cp_upper(k, n, conf):
    try:
        from scipy.stats import beta
        return 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))
    except Exception:
        return min(1.0, k / n + math.sqrt(math.log(1.0 / (1.0 - conf)) / (2 * n)))


@torch.no_grad()
def collect(model, ids, dev, layers, n):
    cap = {li: [] for li in layers}
    hs = []
    for li in layers:
        def mk(li):
            def hk(_m, a):
                cap[li].append(a[0].detach().reshape(-1, a[0].shape[-1]))
            return hk
        hs.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    model(ids[:n].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    return {li: torch.cat(cap[li]).float() for li in layers}


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0].to(dev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers
    nL = len(layers); Ls, Lpp, Lppp = silu_consts(dev)
    H = collect(model, ids, dev, list(range(nL)), N_TOK)
    T = H[0].shape[0]

    for b in BITS:
        # token-ball certified layer deviation = max over layers of (max reachable D_e)
        worst_layer_dev = torch.zeros(T, device=dev)   # max over layers per token
        for li in range(nL):
            mlp = layers[li].mlp
            Wg = mlp.gate.weight.float(); rown = Wg.norm(dim=1)
            Hi = H[li]
            lg = Hi @ Wg.T                              # [T,N]
            K, N = mlp.top_k, mlp.num_experts
            ub = lg + EPS * rown; lb = lg - EPS * rown
            # reachable[t,e] = e could be in top-K over the ball
            # e NOT reachable iff >=K experts have lb_j > ub_e
            reach = torch.zeros(T, N, dtype=torch.bool, device=dev)
            for e in range(N):
                cnt = (lb > ub[:, e:e+1]).sum(1)
                reach[:, e] = cnt < K
            # per-expert spectral consts (cache) and bound per token where reachable
            for e in range(N):
                te = reach[:, e].nonzero().flatten()
                if len(te) == 0:
                    continue
                W1 = mlp.experts[e].gate_proj.weight.float()
                W2 = mlp.experts[e].up_proj.weight.float()
                W3 = mlp.experts[e].down_proj.weight.float()
                W1q, W2q, W3q = quant(W1, b), quant(W2, b), quant(W3, b)
                sv = (spec(W1), spec(W2), spec(W1q), spec(W2q), spec(W1-W1q), spec(W2-W2q))
                for t in te.tolist():
                    d = expert_bound(W1, W2, W3, W1q, W2q, W3q, Hi[t], EPS, sv, Ls, Lpp, Lppp)
                    if d > worst_layer_dev[t]:
                        worst_layer_dev[t] = d
            print(f"  [{b}-bit] layer {li} done", flush=True)

        print(f"\n{'='*70}\n  HYBRID certificate ({b}-bit, eps-BALL, deploy n={T}, "
              f"{int(CONF*100)}% conf)\n{'='*70}")
        print(f"  {'delta_layer':>11s} | {'%balls certified':>16s} | {'CP cert: %% uncertified <=':>26s}")
        print("  " + "-"*60)
        for dl in DELTAS:
            ok = int((worst_layer_dev <= dl).sum().item())
            kbad = T - ok
            print(f"  {dl:>11.2f} | {ok/T*100:>15.1f}% | {cp_upper(kbad, T, CONF)*100:>24.1f}%",
                  flush=True)

    print(f"\n  '%balls certified' = deployment tokens whose ENTIRE eps-ball has all-layer")
    print("  compression deviation <= delta (worst-case over the ball, per-component bound).")
    print("  CP = 95%-conf upper bound on the deployment fraction NOT so certified. This is a")
    print("  distributional guarantee over eps-NEIGHBORHOODS (vs exp 98's points) -- the")
    print("  verification engines certify each ball; Clopper-Pearson lifts to the distribution.")


if __name__ == "__main__":
    main()
