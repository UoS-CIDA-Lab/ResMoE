"""Experiment 123 — Gate-sketch routing: can a cheap PHYSICAL measurement beat a learned
predictor, and close the deployable->oracle routing gap that exp 110 declared intrinsic?

The MoE-preserving constraint forces the only real lever to be ROUTING (exp 108): which
groups to compute exactly vs drop-to-representative, decided WITHOUT computing the dropped
activations. exp 110 swept LEARNED routers (MLP on x / PCA(x) / full-x) and found they
PLATEAU well above the oracle ceiling -> "which fine group to drop is intrinsically
unpredictable FROM x". But every variant there *learns a regression from x*. None used a
cheap PHYSICAL sketch of the expert's own weights.

KEY DISTINCTION (the open gap): routing needs the ORDER of per-neuron magnitude, not its
value. exp 121 showed low-rank cannot REPLACE the computation (output lives in the spectral
tail through the nonlinearity); exp 120 found W1's stable rank is very low. These coexist:
low-rank is too coarse to COMPUTE o_g, but may be plenty to RANK which neurons are large.
So we route on a rank-r sketch of the gate/up projections (the optimal linear measurement,
no learning, no calib regression), then compute only the selected groups exactly.

Routers (all score GROUPS -> topk -> unit mask; representative FIXED to STATIC mean to
isolate selection, as in exp 108):
  oracle-resid : exact activation, exact per-group ||dropped contribution||   (CEILING)
  agg-exact    : exact activation, CHEAP per-neuron L2 aggregate (= sketch r=inf)
                 gap to oracle = cost of ignoring within-group cancellation (exp 111)
  sketch-{r}   : rank-r physical sketch of gate+up -> est. per-neuron mag -> cheap aggregate
                 (DEPLOYABLE, ours; cost r*(d+dff), no W3 matmul at routing time)
  mlp-pca      : exp 110's best deployable LEARNED router (MLP on 128 PCA feats)  (BASELINE)

Read: does sketch-{r} climb toward oracle-resid as r grows, and beat mlp-pca at comparable
routing cost -- especially at fine G where learned routers plateaued (exp 110)?

OLMoE experts as dense SwiGLU FFNs, all layer-0 tokens, calib/eval split.
Run: python3 experiments/123_gate_sketch_router.py
"""
from __future__ import annotations

import sys
import pathlib
import statistics as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
LAYER = 0
N_TOK = 2400
N_EXPERTS = 8
GGRID = [16, 128]
KEEP = [0.5, 0.85]
RGRID = [8, 16, 32, 64]      # sketch ranks
RFEAT = 128                  # PCA dims for the learned baseline router
CALIB_FRAC = 0.6


def kmeans(X, k, iters=30):
    c = X[torch.randperm(X.shape[0], device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a


def collect_inputs(model, ids, dev, layer, n):
    cap = []
    h = model.model.layers[layer].mlp.register_forward_pre_hook(
        lambda _m, a: cap.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    with torch.no_grad():
        model(ids[:n].unsqueeze(0).to(dev))
    h.remove()
    return torch.cat(cap).float()


def mlp_fit(X, Y, hidden=128, steps=400, lr=5e-3):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            F.mse_loss(net(X), Y).backward()
            opt.step()
    return net.eval()


def group_resid(dev_act, idx, W3):       # exact ||W3[:,g] @ dev_act[:,g]|| per group -> [n,G]
    return torch.stack([(dev_act[:, ix] @ W3[:, ix].T).norm(dim=1) for ix in idx], 1)


def cheap_agg(imp, idx, G):              # L2-aggregate per-neuron importance per group -> [n,G]
    return torch.stack([imp[:, ix].norm(dim=1) for ix in idx], 1)


def sketch_act(X, W1, W2, r, mean_u):
    """rank-r physical sketch of the SwiGLU hidden activation u = silu(W1 x) * (W2 x)."""
    U1, S1, V1 = torch.linalg.svd(W1, full_matrices=False)   # W1 [dff,d]
    U2, S2, V2 = torch.linalg.svd(W2, full_matrices=False)
    g = ((X @ V1[:r].T) * S1[:r]) @ U1[:, :r].T              # ~ X @ W1_r.T   [n,dff]
    up = ((X @ V2[:r].T) * S2[:r]) @ U2[:, :r].T             # ~ X @ W2_r.T
    return F.silu(g) * up                                    # estimated activation [n,dff]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    H = collect_inputs(model, ids, dev, LAYER, N_TOK)
    mlp = model.model.layers[LAYER].mlp
    d = H.shape[1]
    EW = {e: (mlp.experts[e].gate_proj.weight.detach().float(),
              mlp.experts[e].up_proj.weight.detach().float(),
              mlp.experts[e].down_proj.weight.detach().float())
          for e in range(N_EXPERTS)}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    T = H.shape[0]
    perm = torch.randperm(T, device=dev)
    nc = int(CALIB_FRAC * T)
    Hc, He = H[perm[:nc]], H[perm[nc:]]
    xbar = Hc.mean(0)
    _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
    P = Vt[:RFEAT].T
    Zc, Ze = (Hc - xbar) @ P, (He - xbar) @ P
    ne = He.shape[0]

    # stable rank of gate/up weights (grounds the "low-rank sketch ranks well" hypothesis)
    def stable_rank(W):
        s = torch.linalg.svdvals(W)
        return (s.pow(2).sum() / s[0].pow(2)).item()
    sr_g = st.mean([stable_rank(EW[e][0]) for e in EW])
    sr_u = st.mean([stable_rank(EW[e][1]) for e in EW])
    print(f"gate-sketch router: G={GGRID}, d={d}, dff=1024, {N_EXPERTS} experts; "
          f"calib={len(Hc)} eval={ne}; rep=STATIC")
    print(f"mean stable rank: gate {sr_g:.1f}, up {sr_u:.1f}  (of 1024) "
          f"-> small r should rank neurons well\n")

    routers = (["oracle-resid", "agg-exact", "gate-exact", "gate*meanup"]
               + [f"sketch-{r}" for r in RGRID] + ["mlp-pca"])
    acc = {(rt, kp, G): [] for rt in routers for kp in KEEP for G in GGRID}

    for e in EW:
        W1e, W2e, W3e = EW[e]
        uc = F.silu(Hc @ W1e.T) * (Hc @ W2e.T)
        ue = F.silu(He @ W1e.T) * (He @ W2e.T)
        dff = uc.shape[1]
        mean_u = uc.mean(0)                       # STATIC representative
        Ye = ue @ W3e.T
        yn = Ye.norm(dim=1).clamp(min=1e-8)
        dev_c, dev_e = uc - mean_u, ue - mean_u
        w3n = W3e.norm(dim=0)                      # ||W3[:,j]||, precomputed offline

        # per-neuron importance estimates (|u_j - mean| * ||w3_j||): exact and sketched
        imp_exact = dev_e.abs() * w3n
        imp_sketch = {r: (sketch_act(He, W1e, W2e, r, mean_u) - mean_u).abs() * w3n
                      for r in RGRID}
        # gate-based routers: gate is the natural SwiGLU gating signal (computed anyway),
        # stable rank ~45 << up's ~160. "gate-exact": route on exact silu(gate), skip up+down
        # on dropped neurons -> real compute saving (1+2k)/3, uses EXACT gate.
        g_exact = F.silu(He @ W1e.T)
        mean_up = (Hc @ W2e.T).mean(0)            # static up estimate
        imp_gate = g_exact.abs() * w3n                                  # ignore up entirely
        imp_gate_mu = (g_exact * mean_up).abs() * w3n                   # exact gate x static up

        def err(mask):
            kept = ue * mask + mean_u * (1 - mask)
            return ((kept @ W3e.T - Ye).norm(dim=1) / yn * 100)

        for G in GGRID:
            Xn = W1e / (W1e.norm(dim=1, keepdim=True) + 1e-8)
            grp = kmeans(Xn, G)
            idx = [torch.nonzero(grp == g).flatten() for g in range(G)]
            gr_c = group_resid(dev_c, idx, W3e)
            gr_e = group_resid(dev_e, idx, W3e)
            net = mlp_fit(Zc, gr_c)
            score = {
                "oracle-resid": gr_e,
                "agg-exact": cheap_agg(imp_exact, idx, G),
                "gate-exact": cheap_agg(imp_gate, idx, G),
                "gate*meanup": cheap_agg(imp_gate_mu, idx, G),
                "mlp-pca": net(Ze).detach(),
            }
            for r in RGRID:
                score[f"sketch-{r}"] = cheap_agg(imp_sketch[r], idx, G)

            for kp in KEEP:
                ng = max(1, round(kp * G))
                for rt in routers:
                    keep_g = score[rt].topk(ng, dim=1).indices
                    selg = torch.zeros(ne, G, dtype=torch.bool, device=dev)
                    selg.scatter_(1, keep_g, True)
                    m = torch.zeros(ne, dff, device=dev)
                    for g in range(G):
                        if selg[:, g].any():
                            m[selg[:, g][:, None] & (grp == g)[None, :]] = 1.0
                    acc[(rt, kp, G)] += err(m).tolist()
        print(f"  expert {e} done", flush=True)

    for kp in KEEP:
        print(f"\n=== keep {int(kp*100)}% === rel-L2 FFN-output err vs dense (median %), "
              f"rep=STATIC")
        hdr = f"  {'router':>14s} | " + " | ".join(f"G={G:<4d}" for G in GGRID)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for rt in routers:
            cells = " | ".join(f"{st.median(acc[(rt, kp, G)]):>6.1f}" for G in GGRID)
            print(f"  {rt:>14s} | {cells}")

    print("\nINTERPRET: oracle-resid=ceiling, mlp-pca=exp110 deployable baseline.")
    print(" - sketch-r climbing toward oracle as r grows  => routing info IS in the weights,")
    print("   recoverable by cheap MEASUREMENT (not learnable from x) -> beats exp 110's wall.")
    print(" - agg-exact gap to oracle = within-group cancellation cost (exp 111); shrinks at")
    print("   fine G, so sketch routing should help MOST at large G (where exp 110 plateaued).")
    print(" - sketch-r ~ mlp-pca or worse => measurement no better than learning; honest neg.")


if __name__ == "__main__":
    main()
