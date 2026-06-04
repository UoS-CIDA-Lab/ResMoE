"""Experiment 107 — Improving G-MoEfication's representative value with our
cancellation insight (input-conditioned low-rank correction).

G-MoEfication (Lee et al., EMNLP 2024) MoEfies a non-sparse (GeLU/SwiGLU) FFN by, for
each UNSELECTED expert-group i, substituting a STATIC representative value r_i =
mean(sigma(h_i)) -- a rank-0, input-INDEPENDENT constant (their Eq 16-17). It works on a
forgiving task metric, but throws away the input-dependent part of the dropped
contribution. Our finding (exp 105/106): the dropped part is high-rank input-dependent
sign-cancellation. exp 105 showed an *oracle* in-sample SVD recovers some of it -- but
that is not deployable (no train/test split, coefficients not predicted from input).

Here we make it deployable and test the real question: is the dropped contribution
predictable FROM THE INPUT x? We fit, per group, a low-rank linear map o_g(x) ~
mean_o_g + U_g (P^T(x - xbar)) where P = top-k PCA dirs of the FFN input (SHARED across
groups), U_g by ridge least-squares on a CALIB split, scored on a held-out EVAL split.
The unselected aggregate correction is then (sum mean_o_g) + (sum U_g)(P^T(x-xbar)) -- a
rank-k input-conditioned vector, k*d extra FLOPs (negligible vs the FFN). Schemes:
  ZERO   : naive MoEfication (drop -> 0)
  STATIC : G-MoEfication (drop -> mean_o_g)            [rank-0, input-independent]
  LR-k   : ours (drop -> mean_o_g + U_g P^T(x-xbar))   [rank-k, input-conditioned]
  ORACLE : exp-105-style in-sample SVD of the EVAL residual (upper bound, not deployable)
Both G-MoEfication 'similarity selection' routing and an oracle (true group-norm) routing.

Treats each OLMoE expert as a dense SwiGLU FFN (d=2048 -> dff=1024 -> 2048), evaluated on
ALL layer-0 inputs (the FFN is a function over its whole input domain, independent of
routing). Run: python3 experiments/107_gmoefication_lowrank.py
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
G = 16                       # expert-groups per FFN (G-MoEfication uses K=64 on full FFN)
RETENTION = [0.5, 0.75, 0.85]
KS = [2, 4, 8]               # low-rank widths for our correction
CALIB_FRAC = 0.6


def kmeans(X, k, iters=30):
    g = torch.randperm(X.shape[0])[:k]
    c = X[g].clone()
    a = torch.zeros(X.shape[0], dtype=torch.long)
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
    dff = mlp.experts[0].gate_proj.weight.shape[0]
    experts = list(range(N_EXPERTS))
    EW = {e: (mlp.experts[e].gate_proj.weight.detach().float().cpu(),
              mlp.experts[e].up_proj.weight.detach().float().cpu(),
              mlp.experts[e].down_proj.weight.detach().float().cpu())
          for e in experts}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    T = H.shape[0]
    perm = torch.randperm(T)
    nc = int(CALIB_FRAC * T)
    ci, ei = perm[:nc], perm[nc:]
    Hc, He = H[ci], H[ei]
    print(f"SwiGLU FFN G-MoEfication: dff={dff} units -> G={G} groups, d={d}; "
          f"{len(experts)} experts; tokens calib={len(ci)} eval={len(ei)}\n")

    # accumulate rel-L2 error (%) per (routing, retention, scheme) over experts*eval-tokens
    schemes = ["ZERO", "STATIC", *[f"LR-{k}" for k in KS], "ORACLE"]
    routings = ["similarity", "pred-norm", "oracle"]
    acc = {(rt, r, s): [] for rt in routings for r in RETENTION for s in schemes}

    # shared input PCA (from calib) -- same projection reused for every expert.
    # kmax dims feed the low-rank correction; R(>=kmax) dims feed the router head.
    R = 64
    xbar = Hc.mean(0)
    _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
    P = Vt[:max(KS)].T                                            # [d,kmax]
    Zc = (Hc - xbar) @ P                                          # [nc,kmax]
    Ze = (He - xbar) @ P                                          # [ne,kmax]
    Pr = Vt[:R].T                                                 # [d,R] router feats
    Zrc = (Hc - xbar) @ Pr                                        # [nc,R]
    Zre = (He - xbar) @ Pr                                        # [ne,R]

    for e in experts:
        W1e, W2e, W3e = EW[e]
        # --- group the dff units (G-MoEfication "parameter clustering split" on W1 rows)
        Xn = W1e / (W1e.norm(dim=1, keepdim=True) + 1e-8)         # [dff,d] unit input dirs
        grp = kmeans(Xn, G)
        idx = [torch.nonzero(grp == g).flatten() for g in range(G)]
        rep = torch.stack([Xn[idx[g]].mean(0) for g in range(G)])  # group rep for routing

        def acts(Hx):
            return F.silu(Hx @ W1e.T) * (Hx @ W2e.T)              # [n,dff]
        uc, ue = acts(Hc), acts(He)
        # per-group output contribution o_g [n,d]
        Oc = torch.stack([uc[:, idx[g]] @ W3e[:, idx[g]].T for g in range(G)])  # [G,nc,d]
        Oe = torch.stack([ue[:, idx[g]] @ W3e[:, idx[g]].T for g in range(G)])  # [G,ne,d]
        Yc, Ye = Oc.sum(0), Oe.sum(0)                              # dense FFN output
        mean_o = Oc.mean(1)                                        # [G,d] STATIC rep value

        # --- fit per-group low-rank maps U_g on the SHARED input PCA (calib only)
        # ridge LS: (Oc[g]-mean_o[g]) ~ Zc[:, :k] @ Ug[g]   (Ug[g] is [k,d])
        Dc = Oc - mean_o.unsqueeze(1)                             # [G,nc,d] calib residual
        lam = 1e-3
        Ug = {}
        for k in KS:
            Zk = Zc[:, :k]
            Ainv = torch.linalg.inv(Zk.T @ Zk + lam * torch.eye(k))
            Ug[k] = torch.einsum("ij,gjd->gid", Ainv,
                                 torch.einsum("nk,gnd->gkd", Zk, Dc))  # [G,k,d]

        # --- routing scores on eval tokens
        sim = (He / (He.norm(dim=1, keepdim=True) + 1e-8)) @ rep.T  # [ne,G] cosine
        onorm = Oe.norm(dim=2).T                                    # [ne,G] true group norm
        # pred-norm (ours): ridge regress calib group-contribution-norm on router feats
        ycn = Oc.norm(dim=2).T                                      # [nc,G] calib norms
        Wr = torch.linalg.solve(Zrc.T @ Zrc + 1e-2 * torch.eye(R), Zrc.T @ ycn)  # [R,G]
        pnorm = Zre @ Wr                                            # [ne,G] predicted

        ne = He.shape[0]
        for r in RETENTION:
            n_keep = max(1, round(r * G))
            for rt, score in (("similarity", sim), ("pred-norm", pnorm),
                              ("oracle", onorm)):
                keep = score.topk(n_keep, dim=1).indices            # [ne,n_keep]
                sel = torch.zeros(ne, G, dtype=torch.bool)
                sel.scatter_(1, keep, True)                         # [ne,G]
                unsel = ~sel
                base = torch.einsum("gnd,ng->nd", Oe, sel.float())  # exact selected sum
                # ZERO
                eZ = (base - Ye).norm(dim=1) / Ye.norm(dim=1)
                # STATIC: + sum of mean_o over unselected
                cs = torch.einsum("gd,ng->nd", mean_o, unsel.float())
                eS = (base + cs - Ye).norm(dim=1) / Ye.norm(dim=1)
                acc[(rt, r, "ZERO")] += (eZ * 100).tolist()
                acc[(rt, r, "STATIC")] += (eS * 100).tolist()
                # LR-k: + sum over unselected of (mean_o + U_g Ze)
                for k in KS:
                    pred = mean_o.unsqueeze(0) + torch.einsum("nk,gkd->ngd", Ze[:, :k], Ug[k])  # [ne,G,d]
                    cl = torch.einsum("ngd,ng->nd", pred, unsel.float())
                    eL = (base + cl - Ye).norm(dim=1) / Ye.norm(dim=1)
                    acc[(rt, r, f"LR-{k}")] += (eL * 100).tolist()
                # ORACLE: exp-105-style in-sample SVD of the unselected EVAL residual
                resid = torch.einsum("gnd,ng->nd", Oe, unsel.float())  # true dropped sum
                resid_c = resid - cs                                 # after removing static mean
                Uo, So, Vto = torch.linalg.svd(resid_c, full_matrices=False)
                ro = min(max(KS), So.shape[0])
                approx = cs + (Uo[:, :ro] * So[:ro]) @ Vto[:ro]
                eO = (base + approx - Ye).norm(dim=1) / Ye.norm(dim=1)
                acc[(rt, r, "ORACLE")] += (eO * 100).tolist()

    for rt in routings:
        print(f"=== routing = {rt} === (relative L2 output error vs dense FFN, median %)")
        hdr = f"{'keep':>5s} | " + " | ".join(f"{s:>7s}" for s in schemes)
        print(hdr)
        print("-" * len(hdr))
        for r in RETENTION:
            row = " | ".join(f"{st.median(acc[(rt, r, s)]):>7.1f}" for s in schemes)
            print(f"{int(r*100):>4d}% | {row}")
        print()
    print("STATIC = G-MoEfication (rank-0 input-independent). LR-k = ours (rank-k input-")
    print("conditioned, k*d extra FLOPs). ORACLE = in-sample SVD upper bound (not deployable).")
    print("LR-k << STATIC and close to ORACLE => dropped cancellation IS input-predictable")
    print("=> our low-rank representative beats G-MoEfication's constant. LR-k ~ STATIC =>")
    print("dropped part is input-UNpredictable; only the static mean / fine-tuning helps.")


if __name__ == "__main__":
    main()
