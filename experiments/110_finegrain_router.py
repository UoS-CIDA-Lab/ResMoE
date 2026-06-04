"""Experiment 110 — Closing the fine-granularity router gap (direction 1).

exp 109: at fine G the deployable MLP router falls well short of the oracle ceiling
(G=128, 85% keep: mlp-resid 12.9 vs oracle-resid 6.6). Is that a router-CAPACITY gap
(closable with a bigger/full-input router) or a fundamental unpredictability? We hold
G=128 and the LR representative fixed and sweep router capacity:
  mlp-pca   : 128 PCA feats -> hidden 128 (exp 109's router)
  mlp-pca-big: 128 PCA feats -> hidden 512
  mlp-fullx : full d=2048 input -> hidden 256
  mlp-deep  : 256 PCA feats -> 256 -> 256
  oracle-resid : true per-group residual (ceiling)
All target the per-group post-correction residual ||o_g - rep_g|| (the right objective).

OLMoE experts as dense SwiGLU FFNs, all layer-0 tokens, calib/eval split.
Run: python3 experiments/110_finegrain_router.py
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
G = 128
KEEP = [0.5, 0.85]
KLR = 8
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


def ridge_fit(X, Y, lam=1e-2):
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T @ Y)


def mlp_fit(X, Y, hidden, steps=500, lr=5e-3):
    layers = []
    fin = X.shape[1]
    hs = hidden if isinstance(hidden, (list, tuple)) else [hidden]
    for hh in hs:
        layers += [torch.nn.Linear(fin, hh), torch.nn.Tanh()]
        fin = hh
    layers += [torch.nn.Linear(fin, Y.shape[1])]
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(net(X), Y).backward()
        opt.step()
    return net


def group_resid(dev_act, idx, W3):
    return torch.stack([(dev_act[:, ix] @ W3[:, ix].T).norm(dim=1) for ix in idx], 1)


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
    Hc, He = H[perm[:nc]], H[perm[nc:]]
    xbar = Hc.mean(0)
    _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
    P = Vt[:256].T
    Zc, Ze = (Hc - xbar) @ P, (He - xbar) @ P          # up to 256 PCA feats
    Xc, Xe = Hc - xbar, He - xbar                       # full centered input
    print(f"fine-G router sweep: G={G}, d={d}, {N_EXPERTS} experts; "
          f"calib={len(Hc)} eval={len(He)}; rep=LR-{KLR}\n")

    routers = ["mlp-pca", "mlp-pca-big", "mlp-fullx", "mlp-deep", "oracle-resid"]
    acc = {(rt, kp): [] for rt in routers for kp in KEEP}

    for e in EW:
        W1e, W2e, W3e = EW[e]
        uc = F.silu(Hc @ W1e.T) * (Hc @ W2e.T)
        ue = F.silu(He @ W1e.T) * (He @ W2e.T)
        dff = uc.shape[1]
        mean_u = uc.mean(0)
        Ye = ue @ W3e.T
        yn = Ye.norm(dim=1)
        Wlr = ridge_fit(Zc[:, :KLR], uc - mean_u)
        repLR = mean_u + Ze[:, :KLR] @ Wlr
        dev_c, dev_e = uc - mean_u, ue - mean_u
        ne = He.shape[0]

        Xn = W1e / (W1e.norm(dim=1, keepdim=True) + 1e-8)
        grp = kmeans(Xn, G)
        idx = [torch.nonzero(grp == g).flatten() for g in range(G)]
        gr_c = group_resid(dev_c, idx, W3e)
        gr_e = group_resid(dev_e, idx, W3e)

        nets = {
            "mlp-pca": mlp_fit(Zc[:, :128], gr_c, 128),
            "mlp-pca-big": mlp_fit(Zc[:, :128], gr_c, 512),
            "mlp-fullx": mlp_fit(Xc, gr_c, 256),
            "mlp-deep": mlp_fit(Zc[:, :256], gr_c, [256, 256]),
        }
        score = {"mlp-pca": nets["mlp-pca"](Ze[:, :128]).detach(),
                 "mlp-pca-big": nets["mlp-pca-big"](Ze[:, :128]).detach(),
                 "mlp-fullx": nets["mlp-fullx"](Xe).detach(),
                 "mlp-deep": nets["mlp-deep"](Ze[:, :256]).detach(),
                 "oracle-resid": gr_e}

        for kp in KEEP:
            ng = max(1, round(kp * G))
            for rt in routers:
                keep_g = score[rt].topk(ng, dim=1).indices
                selg = torch.zeros(ne, G, dtype=torch.bool)
                selg.scatter_(1, keep_g, True)
                m = torch.zeros(ne, dff)
                for g in range(G):
                    if selg[:, g].any():
                        m[selg[:, g][:, None] & (grp == g)[None, :]] = 1.0
                kept = ue * m + repLR * (1 - m)
                err = (kept @ W3e.T - Ye).norm(dim=1) / yn * 100
                acc[(rt, kp)] += err.tolist()
        print(f"  expert {e} done", flush=True)

    print()
    for kp in KEEP:
        print(f"=== keep {int(kp*100)}% (G={G}, rep=LR) rel-L2 FFN-output err median %")
        for rt in routers:
            print(f"   {rt:>14s} : {st.median(acc[(rt, kp)]):>6.1f}")
        print()
    print("If bigger/full-input routers approach oracle-resid -> the fine-G gap is router")
    print("CAPACITY (closable). If they plateau well above it -> which fine groups to drop")
    print("is fundamentally hard to predict from x (intrinsic limit of cheap routing).")


if __name__ == "__main__":
    main()
