"""Experiment 108 — Pushing the G-MoEfication router (routing is the dominant lever).

exp 107 showed selection quality dominates the representative-value choice: oracle
group-routing crushed G-MoEfication's input-cosine 'similarity selection' (50% keep:
47% vs 75% rel-L2 FFN error). Here we sweep ROUTERS with the representative FIXED to
G-MoEfication's STATIC mean (so only selection varies), and test two ideas:

  (A) Stronger but still-cheap predictors of group importance, on shared input-PCA feats:
      ridge (linear) and a small MLP (G-MoEfication's actual router is a 2-layer MLP).
  (B) The theoretically-correct selection criterion. With a representative-value
      correction, the group you can safely DROP is not the one with small contribution
      ||o_g(x)|| but the one whose contribution is well-explained by its representative,
      i.e. small POST-CORRECTION residual ||o_g(x) - rep_g||. G-MoEfication ranks by
      (predicted) contribution; ranking by the residual is the right objective and gives
      a lower oracle ceiling -- which a cheap predictor of the residual magnitude chases.

Routers (score = importance to KEEP; top-n kept and computed exactly, rest -> mean):
  similarity   : cos(x, group W1-mean)            [G-MoEfication deployable]
  pred-norm    : ridge feats -> ||o_g(x)||         [predict contribution]
  pred-resid   : ridge feats -> ||o_g(x)-rep_g||   [predict post-correction residual, ours]
  mlp-resid    : small MLP feats -> ||o_g-rep_g||  [nonlinear version]
  oracle-norm  : true ||o_g(x)||                   [G-MoEfication groundtruth target]
  oracle-resid : true ||o_g(x)-rep_g||             [true optimum for the STATIC correction]

OLMoE experts as dense SwiGLU FFNs, all layer-0 tokens, calib/eval split.
Run: python3 experiments/108_gmoefication_router.py
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
G = 16
RETENTION = [0.5, 0.75, 0.85]
RFEAT = 128            # PCA dims used as router features (cheap)
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


def ridge_fit(X, Y, lam=1e-2):                 # X[n,f] Y[n,g] -> W[f,g]
    f = X.shape[1]
    return torch.linalg.solve(X.T @ X + lam * torch.eye(f), X.T @ Y)


def mlp_fit(X, Y, hidden=128, steps=400, lr=5e-3):
    f, g = X.shape[1], Y.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(f, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, g))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(net(X), Y)
        loss.backward()
        opt.step()
    return net


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
    d = H.shape[1]
    EW = {e: (mlp.experts[e].gate_proj.weight.detach().float().cpu(),
              mlp.experts[e].up_proj.weight.detach().float().cpu(),
              mlp.experts[e].down_proj.weight.detach().float().cpu())
          for e in range(N_EXPERTS)}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    T = H.shape[0]
    perm = torch.randperm(T)
    nc = int(CALIB_FRAC * T)
    ci, ei = perm[:nc], perm[nc:]
    Hc, He = H[ci], H[ei]
    xbar = Hc.mean(0)
    _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
    Pr = Vt[:RFEAT].T
    Zrc, Zre = (Hc - xbar) @ Pr, (He - xbar) @ Pr      # router feats [n,RFEAT]
    print(f"router sweep: dff=1024->G={G}, d={d}, {N_EXPERTS} experts; "
          f"calib={len(ci)} eval={len(ei)}; representative=STATIC mean\n")

    routers = ["similarity", "pred-norm", "pred-resid", "mlp-resid",
               "oracle-norm", "oracle-resid"]
    acc = {(rt, r): [] for rt in routers for r in RETENTION}

    for e in EW:
        W1e, W2e, W3e = EW[e]
        Xn = W1e / (W1e.norm(dim=1, keepdim=True) + 1e-8)
        grp = kmeans(Xn, G)
        idx = [torch.nonzero(grp == g).flatten() for g in range(G)]
        rep_dir = torch.stack([Xn[idx[g]].mean(0) for g in range(G)])

        def acts(Hx):
            return F.silu(Hx @ W1e.T) * (Hx @ W2e.T)
        uc, ue = acts(Hc), acts(He)
        Oc = torch.stack([uc[:, idx[g]] @ W3e[:, idx[g]].T for g in range(G)])  # [G,nc,d]
        Oe = torch.stack([ue[:, idx[g]] @ W3e[:, idx[g]].T for g in range(G)])  # [G,ne,d]
        Ye = Oe.sum(0)
        mean_o = Oc.mean(1)                                # [G,d] STATIC representative

        # routing targets on calib (transpose to [n,G])
        norm_c = Oc.norm(dim=2).T                          # ||o_g|| calib
        resid_c = (Oc - mean_o.unsqueeze(1)).norm(dim=2).T  # ||o_g - mean|| calib
        Wn = ridge_fit(Zrc, norm_c)
        Wr = ridge_fit(Zrc, resid_c)
        net = mlp_fit(Zrc, resid_c)

        sim = (He / (He.norm(dim=1, keepdim=True) + 1e-8)) @ rep_dir.T
        scores = {
            "similarity": sim,
            "pred-norm": Zre @ Wn,
            "pred-resid": Zre @ Wr,
            "mlp-resid": net(Zre).detach(),
            "oracle-norm": Oe.norm(dim=2).T,
            "oracle-resid": (Oe - mean_o.unsqueeze(1)).norm(dim=2).T,
        }
        ne = He.shape[0]
        for r in RETENTION:
            n_keep = max(1, round(r * G))
            for rt, score in scores.items():
                keep = score.topk(n_keep, dim=1).indices
                sel = torch.zeros(ne, G, dtype=torch.bool)
                sel.scatter_(1, keep, True)
                base = torch.einsum("gnd,ng->nd", Oe, sel.float())
                cs = torch.einsum("gd,ng->nd", mean_o, (~sel).float())
                err = (base + cs - Ye).norm(dim=1) / Ye.norm(dim=1)
                acc[(rt, r)] += (err * 100).tolist()

    print("relative L2 FFN-output error vs dense (median %), representative = STATIC mean")
    hdr = f"{'keep':>5s} | " + " | ".join(f"{rt:>12s}" for rt in routers)
    print(hdr)
    print("-" * len(hdr))
    for r in RETENTION:
        row = " | ".join(f"{st.median(acc[(rt, r)]):>12.1f}" for rt in routers)
        print(f"{int(r*100):>4d}% | {row}")
    print("\nsimilarity = G-MoEfication deployable router. pred/mlp-* = cheap predictors on")
    print(f"{RFEAT} PCA feats. *-norm rank by contribution; *-resid by post-correction")
    print("residual (ours). oracle-resid < oracle-norm => ranking by the residual is the")
    print("right objective; how close pred/mlp-resid get to oracle-resid = the achievable gain.")


if __name__ == "__main__":
    main()
