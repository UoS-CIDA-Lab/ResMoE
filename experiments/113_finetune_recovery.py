"""Experiment 113 — Does our post-hoc advantage survive fine-tuning? (vs G-MoEfication)

Question: assuming post-hoc fine-tuning, do our levers (finer granularity, input-
conditioned low-rank representative) still beat G-MoEfication, or does fine-tuning
equalize? We use an FFN-level distillation proxy: freeze the (MLP) router, then fine-tune
the retained expert weights (W1,W2,W3) + the representative to minimize ||MoEfied(x) -
dense(x)||^2 on calib, eval on held-out. Both methods get the SAME treatment and budget.
This isolates "does a better init -> better/faster recovery" (caveat: real fine-tuning is
end-to-end task loss; this is the FFN-fidelity analog matching our metric).

Configs (50% keep, frozen MLP router): granularity G in {16,128} x representative
{STATIC (G-MoEfication, learnable rank-0 vector), LR (ours, input-conditioned rank-k)}.
We report eval rel-L2 FFN-output error at fine-tune steps {0,100,400}. If the gap at
step 0 closes by step 400 -> fine-tuning equalizes; if it persists -> our init wins.

OLMoE experts as dense SwiGLU FFNs, all layer-0 tokens. GPU.
Run: python3 experiments/113_finetune_recovery.py
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
N_EXPERTS = 4
KEEP = 0.5
KLR = 8
RFEAT = 128
STEPS = [0, 100, 400]
CONFIGS = [(16, "STATIC"), (16, "LR"), (128, "STATIC"), (128, "LR")]
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
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device),
                              X.T @ Y)


def mlp_fit(X, Y, hidden=128, steps=400, lr=5e-3):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(net(X), Y).backward()
        opt.step()
    return net


def unit_mask(score, grp, ng, G, dff, dev):
    ne = score.shape[0]
    keep_g = score.topk(ng, dim=1).indices
    selg = torch.zeros(ne, G, dtype=torch.bool, device=dev)
    selg.scatter_(1, keep_g, True)
    m = torch.zeros(ne, dff, device=dev)
    for g in range(G):
        if selg[:, g].any():
            m[selg[:, g][:, None] & (grp == g)[None, :]] = 1.0
    return m


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
    EW = {e: (mlp.experts[e].gate_proj.weight.detach().float().clone(),
              mlp.experts[e].up_proj.weight.detach().float().clone(),
              mlp.experts[e].down_proj.weight.detach().float().clone())
          for e in range(N_EXPERTS)}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    T = H.shape[0]
    perm = torch.randperm(T, device=dev)
    ncut = int(CALIB_FRAC * T)
    Hc, He = H[perm[:ncut]], H[perm[ncut:]]
    xbar = Hc.mean(0)
    _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
    P = Vt[:RFEAT].T
    Zc, Ze = (Hc - xbar) @ P, (He - xbar) @ P
    print(f"finetune recovery: keep {int(KEEP*100)}%, {N_EXPERTS} experts; "
          f"calib={len(Hc)} eval={len(He)}; steps={STEPS}\n")

    def group_resid(dev_act, idx, W3):
        return torch.stack([(dev_act[:, ix] @ W3[:, ix].T).norm(dim=1) for ix in idx], 1)

    acc = {(G, rp, s): [] for (G, rp) in CONFIGS for s in STEPS}

    for e in EW:
        W1_0, W2_0, W3_0 = EW[e]
        dff = W1_0.shape[0]
        uc0 = F.silu(Hc @ W1_0.T) * (Hc @ W2_0.T)
        ue0 = F.silu(He @ W1_0.T) * (He @ W2_0.T)
        Yc = uc0 @ W3_0.T
        Ye = ue0 @ W3_0.T
        ync, yne = Yc.norm(dim=1), Ye.norm(dim=1)
        mean_u0 = uc0.mean(0)
        Wlr0 = ridge_fit(Zc[:, :KLR], uc0 - mean_u0)

        for (G, rp) in CONFIGS:
            ng = max(1, round(KEEP * G))
            Xn = W1_0 / (W1_0.norm(dim=1, keepdim=True) + 1e-8)
            grp = kmeans(Xn, G)
            idx = [torch.nonzero(grp == g).flatten() for g in range(G)]
            net = mlp_fit(Zc, group_resid(uc0 - mean_u0, idx, W3_0))
            mc = unit_mask(net(Zc).detach(), grp, ng, G, dff, dev)
            me = unit_mask(net(Ze).detach(), grp, ng, G, dff, dev)

            # trainable copies + representative params
            W1 = W1_0.clone().requires_grad_(True)
            W2 = W2_0.clone().requires_grad_(True)
            W3 = W3_0.clone().requires_grad_(True)
            if rp == "STATIC":
                rvec = mean_u0.clone().requires_grad_(True)
                rep_params = [rvec]
            else:
                rmean = mean_u0.clone().requires_grad_(True)
                rW = Wlr0.clone().requires_grad_(True)
                rep_params = [rmean, rW]

            def rep_of(Z):
                if rp == "STATIC":
                    return rvec.unsqueeze(0)
                return rmean.unsqueeze(0) + Z[:, :KLR] @ rW

            def eval_err():
                with torch.no_grad():
                    ue = F.silu(He @ W1.T) * (He @ W2.T)
                    kept = ue * me + rep_of(Ze) * (1 - me)
                    return ((kept @ W3.T - Ye).norm(dim=1) / yne * 100)

            opt = torch.optim.Adam([W1, W2, W3] + rep_params, lr=1e-3)
            s_prev = 0
            for s in STEPS:
                for _ in range(s - s_prev):
                    opt.zero_grad()
                    uc = F.silu(Hc @ W1.T) * (Hc @ W2.T)
                    kept = uc * mc + rep_of(Zc) * (1 - mc)
                    F.mse_loss(kept @ W3.T, Yc).backward()
                    opt.step()
                s_prev = s
                acc[(G, rp, s)] += eval_err().tolist()
        print(f"  expert {e} done", flush=True)

    print(f"\n=== keep {int(KEEP*100)}% rel-L2 FFN-output error vs dense (median %) ===")
    hdr = f"  {'config':>14s} | " + " | ".join(f"step {s:>4d}" for s in STEPS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (G, rp) in CONFIGS:
        cells = " | ".join(f"{st.median(acc[(G, rp, s)]):>9.1f}" for s in STEPS)
        print(f"  {'G=%d/%s' % (G, rp):>14s} | {cells}")
    print("\nstep 0 = post-hoc (no fine-tune). If the G=128/LR vs G=16/STATIC gap shrinks")
    print("toward step 400 -> fine-tuning equalizes our levers; if it persists -> a better")
    print("post-hoc init yields a better post-fine-tune endpoint (our advantage survives).")


if __name__ == "__main__":
    main()
