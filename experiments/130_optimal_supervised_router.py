"""Experiment 130 — Can a deployable router, supervised on the BRUTE-FORCE OPTIMAL
keep-set, beat greedy at AGGRESSIVE ratios? (user idea #2)

Switch (exp 29) showed greedy << optimal at aggressive ratios; never measured for
OLMoE. G=16 groups makes per-token brute-force optimal tractable (quadratic form
e^2(D)=1_D^T Gram_t 1_D over all C(16,drop) drop-sets, Gram_t[i,j]=<r_i(t),r_j(t)>).

Answers:
 (1) SEPARABILITY: is greedy-drop (smallest ||r_g||) == brute optimal? (exp 111 says
     residuals near-orthogonal -> objective separable -> greedy optimal -> MLP+topk is
     the RIGHT structure, no set-arch needed). Measure greedy==opt freq + error gap.
 (2) HEADROOM: greedy vs optimal error at keep 0.25/0.5.
 (3) DEPLOYABLE: train MLP routers on (a) regression to ||r_g||, (b) classification to
     the OPTIMAL keep-set (BCE). Compare eval error to greedy / oracle / optimal.
Run: python3 experiments/130_optimal_supervised_router.py
"""
from __future__ import annotations
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_TOK = 2048
N_EXP = 16          # experts to average over
G = 16              # groups (brute-forceable)
KEEPS = [0.25, 0.5]
SPLIT = 1536        # train / eval


def kmeans(X, k, iters=15, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a


def mlp_fit(X, Y, steps=400, lr=5e-3, bce=False, hidden=128):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = F.binary_cross_entropy_with_logits if bce else F.mse_loss
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            lossf(net(X), Y).backward()
            opt.step()
    return net.eval()


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0][:N_TOK]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)
    mlp0 = model.model.layers[0].mlp
    cap = []
    h = mlp0.register_forward_pre_hook(lambda _m, a: cap.append(a[0].reshape(-1, a[0].shape[-1]).float()))
    model(ids.unsqueeze(0).to(dev))
    h.remove()
    H = torch.cat(cap)                    # [T, d]
    T = H.shape[0]
    print(f"harvested H={tuple(H.shape)}", flush=True)

    # PCA features for routers
    xbar = H.mean(0)
    _, _, Vt = torch.linalg.svd(H - xbar, full_matrices=False)
    P = Vt[:128].T
    Z = (H - xbar) @ P                     # [T,128]

    # enumerate drop-set masks per keep level
    masks = {}
    for keep in KEEPS:
        k = int(round(keep * G)); drop = G - k
        combs = list(itertools.combinations(range(G), drop))
        M = torch.zeros(len(combs), G, device=dev)
        for i, cmb in enumerate(combs):
            M[i, list(cmb)] = 1.0
        masks[keep] = M                    # [#sets, G] drop indicator

    agg = {keep: {kk: [] for kk in ["greedy", "optimal", "oracle", "reg", "cls",
                                     "sep_freq", "y_train"]} for keep in KEEPS}
    for e in range(N_EXP):
        ex = mlp0.experts[e]
        wg = ex.gate_proj.weight.float(); wu = ex.up_proj.weight.float()
        wd = ex.down_proj.weight.float()
        a = F.silu(H @ wg.T) * (H @ wu.T)          # [T,dff]
        abar = a.mean(0); dev_a = a - abar
        grp = kmeans(F.normalize(wg, dim=1), G, seed=e)
        # per-group output residual contributions r_g(t): [T,G,d]
        R = torch.stack([dev_a[:, grp == g] @ wd[:, grp == g].T for g in range(G)], 1)
        y = a @ wd.T                                # [T,d] full output
        ynorm = y.norm(dim=1) + 1e-6
        Gram = torch.einsum('tgd,thd->tgh', R, R)   # [T,G,G]
        rnorm = torch.diagonal(Gram, dim1=1, dim2=2).clamp_min(0).sqrt()  # ||r_g|| [T,G]

        for keep in KEEPS:
            M = masks[keep]; drop = G - int(round(keep * G))
            # brute optimal drop-set per token: e2[t,m] = 1_D^T Gram_t 1_D
            best_err = torch.empty(T, device=dev); best_idx = torch.empty(T, dtype=torch.long, device=dev)
            CH = 4096
            cur = torch.full((T,), float('inf'), device=dev)
            for c0 in range(0, M.shape[0], CH):
                Mc = M[c0:c0+CH]
                e2 = torch.einsum('mi,tij,mj->tm', Mc, Gram, Mc).clamp_min(0)
                v, j = e2.min(1)
                upd = v < cur
                cur = torch.where(upd, v, cur)
                best_idx = torch.where(upd, j + c0, best_idx)
            # recover optimal drop mask per token
            opt_drop = M[best_idx]                      # [T,G]
            opt_err = (cur.sqrt() / ynorm)
            # greedy: drop the `drop` smallest ||r_g||
            gd = rnorm.argsort(1)[:, :drop]
            gmask = torch.zeros(T, G, device=dev); gmask.scatter_(1, gd, 1.0)
            grd_err = (torch.einsum('ti,tij,tj->t', gmask, Gram, gmask).clamp_min(0).sqrt() / ynorm)
            sep = (gmask == opt_drop).all(1).float().mean()    # greedy==optimal?

            # targets for routers: keep-set = 1 - drop
            keep_opt = 1.0 - opt_drop                          # [T,G] optimal keep (k-hot)
            tr, ev = slice(0, SPLIT), slice(SPLIT, T)
            # (a) regression to ||r_g||, top-k
            reg = mlp_fit(Z[tr], rnorm[tr])
            sreg = reg(Z[ev])
            # (b) classification to optimal keep-set, top-k
            cls = mlp_fit(Z[tr], keep_opt[tr], bce=True)
            scls = cls(Z[ev])

            def err_from_keepscore(score):
                k = G - drop
                topk = score.argsort(1, descending=True)[:, :k]
                km = torch.zeros(score.shape[0], G, device=dev); km.scatter_(1, topk, 1.0)
                dm = 1.0 - km
                Gv = Gram[ev]
                return (torch.einsum('ti,tij,tj->t', dm, Gv, dm).clamp_min(0).sqrt()
                        / ynorm[ev]).mean().item()

            agg[keep]["greedy"].append(grd_err[ev].mean().item())
            agg[keep]["optimal"].append(opt_err[ev].mean().item())
            agg[keep]["oracle"].append(err_from_keepscore(rnorm[ev]))     # top-k of TRUE ||r_g||
            agg[keep]["reg"].append(err_from_keepscore(sreg))
            agg[keep]["cls"].append(err_from_keepscore(scls))
            agg[keep]["sep_freq"].append(sep.item())
        print(f"  expert {e} done", flush=True)

    print(f"\nOLMoE layer-0, G={G}, {N_EXP} experts, eval split, rel-L2 FFN-output error\n")
    print(f"  {'keep':>5s} | {'greedy':>7s} | {'optimal':>7s} | {'oracle':>7s} | "
          f"{'reg-MLP':>7s} | {'cls-MLP':>7s} | greedy==opt%")
    print("  " + "-" * 74)
    for keep in KEEPS:
        r = agg[keep]
        m = lambda k: np.mean(r[k]) * 100
        print(f"  {int(keep*100):>4d}% | {m('greedy'):6.1f}% | {m('optimal'):6.1f}% | "
              f"{m('oracle'):6.1f}% | {m('reg'):6.1f}% | {m('cls'):6.1f}% | "
              f"{np.mean(r['sep_freq'])*100:5.1f}%", flush=True)
    print("\nREAD: optimal << greedy => headroom (set structure matters, like Switch exp29).")
    print("      greedy==opt% high => separable => MLP+topk is the right structure (no set-arch).")
    print("      cls < reg => classifying the optimal keep-SET beats regressing scores.")
    print("      best deployable (reg/cls) vs greedy => can supervision beat greedy deployably.")


if __name__ == "__main__":
    main()
