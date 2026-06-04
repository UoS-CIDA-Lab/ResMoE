"""Experiment 125 — Input-region MoE: can STATIC analysis (no fine-tune/distill) decompose
a gated FFN into MoE form by exploiting INPUT-space structure instead of (nonexistent)
neuron-group structure?

Every prior lever decomposed along the NEURON axis (MoEfication's way) and hit the
cancellation wall: the latent neuron-group structure doesn't exist for gated FFNs (exp 114
random≈best, 111 orthogonal high-rank residuals, 121 global low-rank fails). This is the
one untested static axis: cluster the INPUTS into regions (kmeans on calib, no gradient),
and for each region fit a CHEAP local expert (low-rank AFFINE map o≈A_c x+b_c, closed-form
ridge then SVD-truncate to rank r). Route by nearest centroid; each token runs ONE cheap
expert. This is a legitimate MoE form (input-conditioned expert selection) that bypasses
the FFN's internal structure entirely.

HYPOTHESIS: globally the FFN map is high-rank/nonlinear (exp 121: global low-rank fails),
but LOCALLY (within a small input region) it may be ~linear/low-rank (Taylor), so local
experts could be accurate where global ones failed. Test: at matched expert rank r, does
error DROP as the number of regions C grows (C=1 == global linear)? Honest risk (exp 119):
local fits are data-starved and the calib->eval centroid routing can shift -> we use a
calib/eval split and enough tokens so a positive isn't an in-sample artifact.

Metric = rel-L2 FFN-output error vs dense (same as exp 107-123, directly comparable:
G-MoEfication there is ~38% @ eff .5, ~20% @ eff .85). OLMoE experts as dense SwiGLU.
Run: python3 experiments/125_input_region_moe.py
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
N_TOK = 8000
N_EXPERTS = 4
PCA = 256                       # input PCA dim (shared router features + local-model input)
CGRID = [1, 4, 16, 64]          # number of input regions (C=1 == global linear)
RGRID = [64, 128, 256]          # local-expert rank
CALIB_FRAC = 0.6
RIDGE = 1e-1


def kmeans(X, k, iters=30):
    if k == 1:
        return torch.zeros(X.shape[0], dtype=torch.long, device=X.device), X.mean(0, keepdim=True)
    c = X[torch.randperm(X.shape[0], device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return torch.cdist(X, c).argmin(1), c


def collect_inputs(model, ids, dev, layer, n, win=2048):
    cap = []
    h = model.model.layers[layer].mlp.register_forward_pre_hook(
        lambda _m, a: cap.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    got = 0
    with torch.no_grad():
        for s in range(0, ids.shape[0] - 1, win):
            model(ids[s:s + win].unsqueeze(0).to(dev))
            got += min(win, ids.shape[0] - s)
            if got >= n:
                break
    h.remove()
    return torch.cat(cap)[:n].float()


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
    P = Vt[:PCA].T
    Zc, Ze = (Hc - xbar) @ P, (He - xbar) @ P                # [n,PCA]
    ne = He.shape[0]

    def eff(C, r):                                            # frac of dense FFN compute
        return (PCA * d + C * PCA + r * (d + PCA)) / (3 * d * 1024)

    print(f"input-region MoE: layer {LAYER}, {N_EXPERTS} experts as dense SwiGLU, "
          f"d={d}, dff=1024")
    print(f"calib={len(Hc)} eval={ne}, PCA={PCA}; local affine experts, "
          f"closed-form ridge + SVD-rank-r, nearest-centroid routing\n")

    acc = {(C, r): [] for C in CGRID for r in RGRID}
    full_lin = {C: [] for C in CGRID}        # per-region FULL-rank linear ceiling (r=PCA)

    for e in EW:
        W1e, W2e, W3e = EW[e]
        oc = (F.silu(Hc @ W1e.T) * (Hc @ W2e.T)) @ W3e.T     # dense output [nc,d]
        oe = (F.silu(He @ W1e.T) * (He @ W2e.T)) @ W3e.T     # [ne,d]
        on = oe.norm(dim=1).clamp(min=1e-8)

        for C in CGRID:
            grp_c, cent = kmeans(Zc, C)
            grp_e = torch.cdist(Ze, cent).argmin(1)
            # full-rank local linear ceiling + low-rank truncations, per cluster
            pred = {r: torch.zeros(ne, d, device=dev) for r in RGRID}
            pred_full = torch.zeros(ne, d, device=dev)
            for c in range(C):
                ic = (grp_c == c).nonzero().flatten()
                ie = (grp_e == c).nonzero().flatten()
                if ie.numel() == 0:
                    continue
                if ic.numel() < PCA + 2:                     # too few to fit -> use global mean
                    pred_full[ie] = oc.mean(0)
                    for r in RGRID:
                        pred[r][ie] = oc.mean(0)
                    continue
                zc, oc_c = Zc[ic], oc[ic]
                zbar, obar = zc.mean(0), oc_c.mean(0)
                zc0, oc0 = zc - zbar, oc_c - obar
                A = torch.linalg.solve(
                    zc0.T @ zc0 + RIDGE * torch.eye(PCA, device=dev), zc0.T @ oc0)  # [PCA,d]
                ze0 = Ze[ie] - zbar
                pred_full[ie] = obar + ze0 @ A
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)   # rank-r truncation
                for r in RGRID:
                    rr = min(r, S.shape[0])
                    Ar = (U[:, :rr] * S[:rr]) @ Vh[:rr]
                    pred[r][ie] = obar + ze0 @ Ar
            full_lin[C] += ((pred_full - oe).norm(dim=1) / on * 100).tolist()
            for r in RGRID:
                acc[(C, r)] += ((pred[r] - oe).norm(dim=1) / on * 100).tolist()
        print(f"  expert {e} done", flush=True)

    print(f"\nrel-L2 FFN-output err vs dense (median %), by #regions C x expert-rank r:")
    print(f"  (full-rank local-linear ceiling per C in [], eff=frac dense compute)\n")
    hdr = f"  {'C':>4s} {'full[]':>8s} | " + " | ".join(
        f"r={r:<4d}(eff{eff(CGRID[-1], r):.2f})" for r in RGRID)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for C in CGRID:
        cells = " | ".join(f"{st.median(acc[(C, r)]):>14.1f}" for r in RGRID)
        tag = "global" if C == 1 else f"{C} regions"
        print(f"  {C:>4d} {st.median(full_lin[C]):>7.1f} | {cells}   <- {tag}")
    print(f"\n  (G-MoEfication reference, same metric: ~38% @ eff .5, ~20% @ eff .85)")
    print("\nREAD: error DROPS as C grows at fixed r => FFN is locally low-rank-linear;")
    print("input-region static decomposition IS viable -> the door is open. error FLAT")
    print("in C (~= global) => locally just as high-rank; input-region adds nothing, the")
    print("static-decomposition ceiling is closed on this axis too.")


if __name__ == "__main__":
    main()
