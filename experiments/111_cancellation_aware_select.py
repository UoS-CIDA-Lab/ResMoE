"""Experiment 111 — Cancellation-aware selection (direction 2): is the oracle floor
itself lowerable by exploiting cross-group cancellation?

Every router so far (incl. oracle-resid) ranks groups INDEPENDENTLY by their own
post-correction residual magnitude ||r_g||, where r_g(x) = o_g(x) - rep_g is the vector
a dropped group contributes in error. But the actual output error is ||SUM_{g dropped}
r_g|| -- if the dropped groups' residual VECTORS cancel, the true error is far below the
independent-ranking estimate. We test the true objective with an oracle greedy that
builds the drop-set of size n_drop = G - n_keep to minimize the running ||sum|| (at each
step add the group that keeps the accumulated dropped-residual smallest -- it will prefer
groups that cancel what's already dropped). Compare:
  random        : random drop-set (spread reference)
  oracle-resid  : drop the n_drop groups of smallest ||r_g|| (independent ranking)
  oracle-cancel : greedy min-||sum|| drop-set (cancellation-aware, ours)
If oracle-cancel << oracle-resid -> cross-group cancellation is real structure left on
the table by independent ranking (a lower achievable floor, and a target for the
certificate even if hard to route deployably). If ~equal -> residuals are near-orthogonal
across groups; independent ranking is already near-optimal.

G=128, representative=LR-8. OLMoE experts as dense SwiGLU FFNs, all layer-0 tokens.
Run: python3 experiments/111_cancellation_aware_select.py
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


def greedy_cancel(r, rnorm2, n_drop):
    # r [ne,G,d], rnorm2 [ne,G]; build size-n_drop drop set minimizing ||running sum||
    ne, Gn, d = r.shape
    ar = torch.arange(ne)
    S = torch.zeros(ne, d)
    dropped = torch.zeros(ne, Gn, dtype=torch.bool)
    for _ in range(n_drop):
        dot = torch.einsum("nd,ngd->ng", S, r)
        crit = 2 * dot + rnorm2                 # delta ||S||^2 from adding group g
        crit[dropped] = float("inf")
        pick = crit.argmin(1)
        dropped[ar, pick] = True
        S = S + r[ar, pick]
    return S                                     # final dropped-residual sum [ne,d]


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
    nc = int(CALIB_FRAC * T)
    Hc, He = H[perm[:nc]], H[perm[nc:]]
    xbar = Hc.mean(0)
    _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
    P = Vt[:KLR].T
    Zc, Ze = (Hc - xbar) @ P, (He - xbar) @ P
    print(f"cancellation-aware selection: G={G}, {N_EXPERTS} experts; "
          f"calib={len(Hc)} eval={len(He)}; rep=LR-{KLR}\n")

    methods = ["random", "oracle-resid", "oracle-cancel"]
    acc = {(m, kp): [] for m in methods for kp in KEEP}
    # diagnostic: how much do dropped residuals cancel?  ||sum|| / sqrt(sum ||.||^2)
    cancel_ratio = {kp: [] for kp in KEEP}

    for e in EW:
        W1e, W2e, W3e = EW[e]
        uc = F.silu(Hc @ W1e.T) * (Hc @ W2e.T)
        ue = F.silu(He @ W1e.T) * (He @ W2e.T)
        dff = uc.shape[1]
        mean_u = uc.mean(0)
        Wlr = ridge_fit(Zc[:, :KLR], uc - mean_u)
        repLR = mean_u + Ze[:, :KLR] @ Wlr
        dev_e = ue - repLR                          # per-unit residual to dense
        Ye = ue @ W3e.T
        yn = Ye.norm(dim=1)
        ne = He.shape[0]

        Xn = W1e / (W1e.norm(dim=1, keepdim=True) + 1e-8)
        grp = kmeans(Xn, G)
        idx = [torch.nonzero(grp == g).flatten() for g in range(G)]
        # per-group residual VECTORS r_g [ne,G,d]
        r = torch.stack([dev_e[:, ix] @ W3e[:, ix].T for ix in idx], 1)  # [ne,G,d]
        rnorm2 = r.pow(2).sum(-1)                    # [ne,G]

        for kp in KEEP:
            n_drop = max(1, round((1 - kp) * G))
            # random
            ridx = torch.argsort(torch.rand(ne, G), dim=1)[:, :n_drop]
            Sr = torch.gather(r, 1, ridx.unsqueeze(-1).expand(-1, -1, r.shape[-1])).sum(1)
            acc[("random", kp)] += (Sr.norm(dim=1) / yn * 100).tolist()
            # oracle-resid: drop smallest ||r_g||
            sidx = rnorm2.topk(n_drop, dim=1, largest=False).indices
            Ss = torch.gather(r, 1, sidx.unsqueeze(-1).expand(-1, -1, r.shape[-1])).sum(1)
            acc[("oracle-resid", kp)] += (Ss.norm(dim=1) / yn * 100).tolist()
            # diagnostic cancel ratio on the oracle-resid drop set
            indiv = torch.gather(rnorm2, 1, sidx).sum(1).sqrt()
            cancel_ratio[kp] += (Ss.norm(dim=1) / (indiv + 1e-8)).tolist()
            # oracle-cancel: greedy min-||sum||
            Sc = greedy_cancel(r, rnorm2, n_drop)
            acc[("oracle-cancel", kp)] += (Sc.norm(dim=1) / yn * 100).tolist()
        del r, rnorm2
        print(f"  expert {e} done", flush=True)

    print()
    for kp in KEEP:
        print(f"=== keep {int(kp*100)}% (G={G}, rep=LR) rel-L2 FFN-output err median %")
        for m in methods:
            print(f"   {m:>14s} : {st.median(acc[(m, kp)]):>6.1f}")
        print(f"   cancel-ratio on oracle-resid drop set (||sum||/sqrt(sum||.||^2)), "
              f"median: {st.median(cancel_ratio[kp]):.3f}  "
              f"(1=orthogonal, <1=cancellation)")
        print()
    print("oracle-cancel << oracle-resid => cross-group cancellation is exploitable")
    print("structure (lower achievable floor). ~equal => residual vectors are near-")
    print("orthogonal across groups and independent ranking is already near-optimal.")


if __name__ == "__main__":
    main()
