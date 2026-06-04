"""Experiment 76 — Router-conditional tight composition: is the MoE sublayer SMOOTH
inside a certified-routing-stable ball (so the bound composes tightly cell-by-cell),
and how many inputs are certified-stable under real compression drift (coverage)?

Motivation (the router-as-ReLU-pattern reframe): exp 75 showed the per-layer worst-
case Lipschitz is huge (30-inf) BECAUSE small perturbations flip the top-K (a
discontinuity). The router is linear, so the EXACT min-norm flip radius r*(h) is
closed-form (exp 65). Claim: within B2(h, r*) the top-K set is fixed, the MoE sublayer
is SMOOTH, so its Lipschitz collapses to ~1-few (and the compression-difference bound
composes tightly); the blow-up is ENTIRELY the routing flips. If so, MoE verification
should be done cell-by-cell with the router certificate as the (exact) case-split.

We measure, on the live OLMoE MoE sublayer (per token, per layer):
  (1) r*(h) = exact L2 routing-stable radius = min_{s in S, u not in S} (l_s-l_u)/||w_s-w_u||_2.
  (2) UNCONSTRAINED Lipschitz: max ||MoE(h+d)-MoE(h)||/||d|| over ||d|| <= R_BIG (flips allowed).
  (3) CONSTRAINED Lipschitz: same over ||d|| <= 0.99*r* (NO flip) -- should be ~1-few.
  (4) COVERAGE: propagate the real quant-8bit drift; fraction of (token,layer) whose
      drift stays below r* (certified routing-stable under the actual compression).

GPU, live OLMoE. Run: python3 experiments/76_router_conditional.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 64
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14]
N_TOK_LIP = 10            # tokens per layer for the Lipschitz probe
PGD_STEPS = 30
R_BIG = 0.5              # unconstrained ball radius (L2), as in exp 75
PROMPT = ("The mitochondria is the powerhouse of the cell, and researchers have long "
          "studied how energy production scales with demand across tissues and species, "
          "from the smallest insects to the largest whales, under varying conditions.")


def quant8(W):
    qmax = 2 ** 7 - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    nL = len(layers)
    ids = tok(PROMPT, return_tensors="pt").input_ids[:, :N_CALIB].to(dev)

    # cache each MoE sublayer's INPUT (post-attention-norm hidden state the router sees)
    cap = [None] * nL
    hs = []
    for li in range(nL):
        def mk(li):
            def hk(_m, a):
                cap[li] = a[0].detach()       # [1, seq, d]
            return hk
        hs.append(layers[li].mlp.register_forward_pre_hook(mk(li)))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    Hmoe = [cap[li] for li in range(nL)]

    def rstar(mlp, h):
        """exact L2 routing-stable radius at token h [d] (top-K set invariance)."""
        Wg = mlp.gate.weight.float()                 # [N, d]
        lg = Wg @ h.float()                          # [N]
        K = mlp.top_k
        topv, topi = lg.topk(K)
        S = set(topi.tolist())
        sel = torch.tensor(sorted(S), device=h.device)
        uns = torch.tensor([i for i in range(Wg.shape[0]) if i not in S], device=h.device)
        best = float("inf")
        for s in sel:
            for u in uns:
                gap = (lg[s] - lg[u]).item()
                den = (Wg[s] - Wg[u]).norm().item()
                if den > 1e-9:
                    best = min(best, gap / den)
        return best

    def moe_fwd(mlp, h):
        """MoE sublayer on a single token h [d] -> [d] (float)."""
        out = mlp(h.view(1, 1, -1))
        out = out[0] if isinstance(out, tuple) else out
        return out.view(-1).float()

    def lip(mlp, h, radius):
        with torch.no_grad():
            base = moe_fwd(mlp, h)
        v = torch.randn_like(h); v = v / v.norm()
        best = 0.0
        for _ in range(PGD_STEPS):
            v = v.detach().requires_grad_(True)
            pert = moe_fwd(mlp, h + radius * v)
            obj = ((pert - base) ** 2).sum()
            best = max(best, obj.detach().item() ** 0.5 / radius)
            g, = torch.autograd.grad(obj, v)
            with torch.no_grad():
                v = g / (g.norm() + 1e-12)
        return best

    print(f"\n{'='*82}\n  Router-conditional Lipschitz of the MoE sublayer "
          f"(constrained to no-flip vs free)\n{'='*82}")
    print(f"  {'layer':>5s} | {'median r*':>9s} | {'Lip free (||d||<=0.5)':>21s} | "
          f"{'Lip no-flip (||d||<r*)':>22s}")
    print("  " + "-"*78)
    lip_free_all, lip_cons_all = [], []
    rstars = {li: [] for li in range(nL)}
    for li in LAYERS:
        mlp = layers[li].mlp
        H = Hmoe[li][0]
        idx = torch.linspace(0, H.shape[0]-1, N_TOK_LIP).long()
        free, cons, rs = [], [], []
        for t in idx.tolist():
            h = H[t]
            r = rstar(mlp, h)
            rs.append(r)
            free.append(lip(mlp, h, R_BIG))
            cons.append(lip(mlp, h, max(0.99 * r, 1e-4)))
        rstars[li] = rs
        import statistics as st
        print(f"  {li:>5d} | {st.median(rs):>9.4f} | "
              f"{st.median(free):>10.2f} (max {max(free):>7.1f}) | "
              f"{st.median(cons):>10.3f} (max {max(cons):>6.2f})", flush=True)
        lip_free_all += free; lip_cons_all += cons

    # ---- (4) coverage under real quant-8bit compression drift ----
    expert_w = [getattr(layers[li].mlp.experts[e], proj).weight
                for li in range(nL) for e in range(layers[li].mlp.num_experts)
                for proj in ("gate_proj", "up_proj", "down_proj")]
    saved = [w.detach().cpu().clone() for w in expert_w]
    with torch.no_grad():
        for w in expert_w:
            w.data.copy_(quant8(w.float()).to(torch.float16))
    capq = [None] * nL
    hs = []
    for li in range(nL):
        def mk(li):
            def hk(_m, a):
                capq[li] = a[0].detach()
            return hk
        hs.append(layers[li].mlp.register_forward_pre_hook(mk(li)))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    with torch.no_grad():
        for w, s in zip(expert_w, saved):
            w.data.copy_(s.to(dev))

    print(f"\n  COVERAGE: fraction of tokens whose quant-8bit drift < r* (routing certified-stable)")
    print(f"  {'layer':>5s} | {'mean drift':>10s} | {'median r*':>9s} | {'% drift < r* (stable)':>21s}")
    print("  " + "-"*64)
    import statistics as st
    for li in LAYERS:
        H = Hmoe[li][0]; Hq = capq[li][0]
        drift = (Hq - H).norm(dim=-1)               # [seq] per-token L2 drift
        # r* for all tokens this layer
        mlp = layers[li].mlp
        rs = torch.tensor([rstar(mlp, H[t]) for t in range(H.shape[0])], device=dev)
        stable = (drift < rs).float().mean().item()
        print(f"  {li:>5d} | {drift.mean().item():>10.4f} | {st.median(rs.tolist()):>9.4f} | "
              f"{stable*100:>19.1f}%", flush=True)

    print(f"\n  Lip free median {st.median(lip_free_all):.1f} (max {max(lip_free_all):.0f}) "
          f"vs Lip no-flip median {st.median(lip_cons_all):.2f} (max {max(lip_cons_all):.2f}).")
    print("  If no-flip Lip ~1-few while free Lip is huge -> the blow-up IS the routing flip;")
    print("  WITHIN a certified-stable cell the bound composes tightly. Coverage % says for")
    print("  how many inputs the real compression stays inside such a cell (the honest scope).")


if __name__ == "__main__":
    main()
