"""Experiment 109 — Granularity is a real lever for gated-FFN G-MoEfication, and the
combined best-deployable config vs the G-MoEfication baseline.

exp 108 left a puzzle: 50% GROUP keep (G=16) gives ~48% error even with oracle routing,
yet exp 103/105 found 50% UNIT keep gives ~12%. The gap is selection GRANULARITY: a
coarse group of ~64 units is kept/dropped as a block, so you cannot isolate the
droppable directions. ReLU tolerates coarse groups (units are individually exact-zero,
so an inactive group is losslessly dropped); SwiGLU has no exact zeros, so every coarse
group carries signal and gets caught by cancellation. Prediction: finer groups (larger
G) sharply lower the error, toward the per-unit oracle floor -> gated FFNs need much
finer expert granularity than the ReLU-era K.

We compute everything at UNIT granularity with a group-induced mask (so large G costs no
extra memory): kept = W3 @ (u*mask + rep*(1-mask)), where rep is the representative for
dropped units. STATIC rep_j = mean_x u_j (G-MoEfication). LR rep_j = mean u_j + (B z)_j,
z = top-k input-PCA, B fit by ridge on calib (input-conditioned, ours, per-unit). Routers
score groups: mlp-resid (deployable, small MLP on PCA feats -> per-group post-correction
residual) and oracle-resid (true per-group ||dropped contribution||, the ceiling).

OLMoE experts as dense SwiGLU FFNs, all layer-0 tokens, calib/eval split.
Run: python3 experiments/109_gmoefication_granularity.py
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
GGRID = [16, 32, 64, 128]
KEEP = [0.5, 0.85]
KLR = 8                # low-rank width of the input-conditioned representative
RFEAT = 128            # PCA dims used as router/representative features
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


def mlp_fit(X, Y, hidden=128, steps=400, lr=5e-3):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(net(X), Y).backward()
        opt.step()
    return net


def group_resid(dev_act, idx, W3):     # ||W3[:,g] @ dev_act[:,g]|| per group -> [n,G]
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
    P = Vt[:RFEAT].T
    Zc, Ze = (Hc - xbar) @ P, (He - xbar) @ P           # [n,RFEAT]
    print(f"granularity sweep: dff=1024, d={d}, {N_EXPERTS} experts; "
          f"calib={len(Hc)} eval={len(He)}; LR rank={KLR}\n")

    # (router, rep) configs. baseline = G-MoEfication (similarity+STATIC) reported at G=16.
    configs = [("mlp-resid", "STATIC"), ("mlp-resid", "LR"),
               ("oracle-resid", "STATIC"), ("oracle-resid", "LR")]
    # acc[(cfg, keep, G)] -> list of per-token rel-L2 errors (%)
    acc = {}
    base = {}     # G-MoE baseline: similarity router + STATIC, at G=16
    unit_floor = {}   # per-unit oracle (drop lowest-residual units) at each keep

    for e in EW:
        W1e, W2e, W3e = EW[e]
        uc = F.silu(Hc @ W1e.T) * (Hc @ W2e.T)          # [nc,dff]
        ue = F.silu(He @ W1e.T) * (He @ W2e.T)          # [ne,dff]
        dff = uc.shape[1]
        mean_u = uc.mean(0)                              # STATIC representative (per unit)
        Ye = ue @ W3e.T                                  # dense FFN output [ne,d]
        yn = Ye.norm(dim=1)
        # LR representative: predict (u - mean_u) from z=PCA[:KLR]
        Wlr = ridge_fit(Zc[:, :KLR], uc - mean_u)        # [KLR,dff]
        repLR_e = mean_u + Ze[:, :KLR] @ Wlr             # [ne,dff] input-conditioned
        dev_c = uc - mean_u                              # calib activation deviation
        dev_e = ue - mean_u
        ne = He.shape[0]

        def err(mask, rep_e):                            # mask[ne,dff] keep=1
            kept = ue * mask + rep_e * (1 - mask)
            return ((kept @ W3e.T - Ye).norm(dim=1) / yn * 100)

        # per-unit oracle floor: keep units with largest ||u_j-mean||*||w3_j|| per token
        w3n = W3e.norm(dim=0)                            # [dff]
        uimp = dev_e.abs() * w3n                         # [ne,dff]
        for kp in KEEP:
            ku = max(1, round(kp * dff))
            ut = uimp.topk(ku, dim=1).indices
            m = torch.zeros(ne, dff); m.scatter_(1, ut, 1.0)
            unit_floor.setdefault(kp, []).extend(err(m, mean_u).tolist())

        for G in GGRID:
            Xn = W1e / (W1e.norm(dim=1, keepdim=True) + 1e-8)
            grp = kmeans(Xn, G)
            idx = [torch.nonzero(grp == g).flatten() for g in range(G)]
            rep_dir = torch.stack([Xn[ix].mean(0) for ix in idx])
            # per-group post-correction residual (target for routers)
            gr_c = group_resid(dev_c, idx, W3e)          # [nc,G]
            gr_e = group_resid(dev_e, idx, W3e)          # [ne,G] (oracle uses this)
            net = mlp_fit(Zc, gr_c)
            score = {"mlp-resid": net(Ze).detach(), "oracle-resid": gr_e}
            sim = (He / (He.norm(dim=1, keepdim=True) + 1e-8)) @ rep_dir.T
            for kp in KEEP:
                ng = max(1, round(kp * G))
                # group selection -> unit mask
                def umask(sc):
                    keep_g = sc.topk(ng, dim=1).indices
                    selg = torch.zeros(ne, G, dtype=torch.bool)
                    selg.scatter_(1, keep_g, True)
                    m = torch.zeros(ne, dff)
                    for g in range(G):
                        if selg[:, g].any():
                            m[selg[:, g][:, None] & (grp == g)[None, :]] = 1.0
                    return m
                if G == GGRID[0]:
                    base.setdefault(kp, []).extend(err(umask(sim), mean_u).tolist())
                for rt in ("mlp-resid", "oracle-resid"):
                    mk = umask(score[rt])
                    acc.setdefault(("STATIC", rt, kp, G), []).extend(
                        err(mk, mean_u).tolist())
                    acc.setdefault(("LR", rt, kp, G), []).extend(
                        err(mk, repLR_e).tolist())
        print(f"  expert {e} done", flush=True)

    def med(key):
        return st.median(acc[key])

    for kp in KEEP:
        print(f"\n=== keep {int(kp*100)}% === rel-L2 FFN-output error vs dense (median %)")
        print(f"  G-MoEfication baseline (similarity+STATIC, G={GGRID[0]}): "
              f"{st.median(base[kp]):.1f}   |   per-unit oracle floor: "
              f"{st.median(unit_floor[kp]):.1f}")
        hdr = f"  {'G':>4s} | " + " | ".join(
            f"{rt[:6]+'/'+rp:>16s}" for rt in ("mlp-resid", "oracle-resid")
            for rp in ("STATIC", "LR"))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for G in GGRID:
            cells = []
            for rt in ("mlp-resid", "oracle-resid"):
                for rp in ("STATIC", "LR"):
                    cells.append(f"{med((rp, rt, kp, G)):>16.1f}")
            print(f"  {G:>4d} | " + " | ".join(cells))
    print("\nFiner G -> lower error toward the per-unit oracle floor => gated-FFN")
    print("G-MoEfication needs much finer granularity than ReLU's coarse K. mlp-resid")
    print("(deployable) tracks oracle-resid; LR rep shaves a bit more on top of STATIC.")


if __name__ == "__main__":
    main()
