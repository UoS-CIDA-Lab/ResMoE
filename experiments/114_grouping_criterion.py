"""Experiment 114 — Does the grouping CRITERION matter, and should it use OUTPUT weights?

G-MoEfication groups units by input-weight (W1) similarity ("parameter clustering"), a
proxy for co-activation. Two open questions:
 (Q1) Is 'cluster similar units' even the right objective? (vs random / data-driven)
 (Q2) Output weights matter: contribution is o_j(x)=silu(w1_j x)(w2_j x)*w3_j -- ||w3_j||
      scales it and w3_j direction decides cancellation. W1-only ignores w2,w3.
We sweep grouping criteria at matched UNIT budget (select groups until ~keep*dff units,
so unbalanced clusters don't cheat), fixed STATIC representative, isolating grouping with
oracle routing (and mlp routing for the deployable view):
  random   : balanced random partition (no grouping signal)
  W1-sim   : k-means on normalized W1 rows           [G-MoEfication, input weights]
  W3-sim   : k-means on normalized W3 columns        [output weights/directions]
  W1+W3    : k-means on [W1-row | W3-col] (both)
  coact    : k-means on per-unit activation profiles a_j over calib  [data, input side]
  contrib  : k-means on per-unit OUTPUT-contribution profiles (a_j*||w3_j||) AND w3 dir
             via [w3-dir | activation-profile] -- the full output-aware, data-driven one

rel-L2 FFN-output error vs dense, lower=better. OLMoE experts as dense SwiGLU FFNs.
Run: python3 experiments/114_grouping_criterion.py
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
G = 64
KEEP = [0.5, 0.85]
RFEAT = 128
CALIB_FRAC = 0.6


def kmeans(X, k, iters=30):
    c = X[torch.randperm(X.shape[0])[:k]].clone()
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
                              torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(net(X), Y).backward()
        opt.step()
    return net


def nrm(X):
    return X / (X.norm(dim=1, keepdim=True) + 1e-8)


def group_resid(dev_act, idx, W3):
    return torch.stack([(dev_act[:, ix] @ W3[:, ix].T).norm(dim=1) for ix in idx], 1)


def sel_by_budget(score, gsz, budget):
    # keep highest-score groups per token until cumulative units >= budget
    ne, Gn = score.shape
    order = score.argsort(dim=1, descending=True)
    so = gsz[order]
    before = so.cumsum(1) - so
    keep_ord = before < budget
    selg = torch.zeros(ne, Gn, dtype=torch.bool)
    selg.scatter_(1, order, keep_ord)
    return selg


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
    H = collect_inputs(model, ids, dev, LAYER, N_TOK).cpu()
    mlp = model.model.layers[LAYER].mlp
    EW = {e: (mlp.experts[e].gate_proj.weight.detach().float().cpu(),
              mlp.experts[e].up_proj.weight.detach().float().cpu(),
              mlp.experts[e].down_proj.weight.detach().float().cpu())
          for e in range(N_EXPERTS)}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    T = H.shape[0]
    perm = torch.randperm(T)
    ncut = int(CALIB_FRAC * T)
    Hc, He = H[perm[:ncut]], H[perm[ncut:]]
    xbar = Hc.mean(0)
    _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
    P = Vt[:RFEAT].T
    Zc, Ze = (Hc - xbar) @ P, (He - xbar) @ P
    print(f"grouping-criterion sweep: G={G}, {N_EXPERTS} experts; "
          f"calib={len(Hc)} eval={len(He)}; rep=STATIC, matched unit budget\n")

    crits = ["random", "W1-sim", "W3-sim", "W1+W3", "coact", "contrib"]
    routers = ["oracle-resid", "mlp-resid"]
    acc = {(c, rt, kp): [] for c in crits for rt in routers for kp in KEEP}

    for e in EW:
        W1e, W2e, W3e = EW[e]
        dff = W1e.shape[0]
        uc = F.silu(Hc @ W1e.T) * (Hc @ W2e.T)
        ue = F.silu(He @ W1e.T) * (He @ W2e.T)
        mean_u = uc.mean(0)
        Ye = ue @ W3e.T
        yn = Ye.norm(dim=1)
        dev_c, dev_e = uc - mean_u, ue - mean_u
        w3col = W3e.T                                     # [dff,d] per-unit output vec
        w3n = w3col.norm(dim=1, keepdim=True)
        # per-unit activation profile (centered, normalized) over calib
        ap = nrm((uc - mean_u).T)                         # [dff,nc]
        feats = {
            "random": torch.randn(dff, 8),
            "W1-sim": nrm(W1e),
            "W3-sim": nrm(w3col),
            "W1+W3": torch.cat([nrm(W1e), nrm(w3col)], 1),
            "coact": ap,
            # output-aware + data: output direction, scaled by contribution magnitude,
            # concatenated with the (magnitude-weighted) activation profile
            "contrib": torch.cat([nrm(w3col), nrm((uc - mean_u).T * w3n)], 1),
        }
        ne = He.shape[0]
        for c in crits:
            grp = kmeans(feats[c], G)
            idx = [torch.nonzero(grp == g).flatten() for g in range(G)]
            gsz = torch.tensor([len(ix) for ix in idx]).float()
            gr_e = group_resid(dev_e, idx, W3e)
            net = mlp_fit(Zc, group_resid(dev_c, idx, W3e))
            score = {"oracle-resid": gr_e, "mlp-resid": net(Ze).detach()}
            for kp in KEEP:
                budget = kp * dff
                for rt in routers:
                    selg = sel_by_budget(score[rt], gsz, budget)
                    m = torch.zeros(ne, dff)
                    for g in range(G):
                        if selg[:, g].any():
                            m[selg[:, g][:, None] & (grp == g)[None, :]] = 1.0
                    kept = ue * m + mean_u.unsqueeze(0) * (1 - m)
                    err = (kept @ W3e.T - Ye).norm(dim=1) / yn * 100
                    acc[(c, rt, kp)] += err.tolist()
        print(f"  expert {e} done", flush=True)

    for kp in KEEP:
        print(f"\n=== keep {int(kp*100)}% (G={G}, rep=STATIC) rel-L2 err median % ===")
        hdr = f"  {'criterion':>10s} | " + " | ".join(f"{rt:>12s}" for rt in routers)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for c in crits:
            row = " | ".join(f"{st.median(acc[(c, rt, kp)]):>12.1f}" for rt in routers)
            print(f"  {c:>10s} | {row}")
    print("\nLower=better. Compare W1-sim (G-MoEfication) vs W3/W1+W3/contrib (output-aware)")
    print("and vs random. If output-aware beats W1-sim -> grouping should use w3; if all ~=")
    print("random -> grouping barely matters (near-orthogonal residuals dominate, exp111).")


if __name__ == "__main__":
    main()
