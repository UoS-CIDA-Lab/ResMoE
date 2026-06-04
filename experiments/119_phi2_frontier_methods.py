"""Experiment 119 — Diverse constructive levers for the COMPUTE-accuracy frontier of
post-hoc G-MoEfication on Phi-2 (faithful lm-eval SuperGLUE subset).

finer-granularity is dead (exp 118), so K=64 fixed. We test levers with a DIFFERENT
mechanism than granularity, all compared at a matched AVERAGE keep (= matched FFN FLOPs):
  uniform      : G-MoEfication baseline (same keep every layer/token, static rep)
  per-layer    : sensitivity-weighted keep across layers (more compute to sensitive
                 layers, less to robust ones); same average
  per-token    : dynamic keep -- drop groups until predicted dropped-residual hits a
                 per-layer threshold; easy tokens keep fewer (lower average FLOPs)
  LR-rep       : input-conditioned low-rank representative for dropped units (carries
                 more info per dropped unit; biggest help when dropping a lot)
  combined     : per-layer + per-token + LR-rep
Goal: at an aggressive average keep (more FLOP savings) does any lever maintain accuracy
better than uniform -> a real frontier push (the technical contribution finer-G is not).

Run: python3 experiments/119_phi2_frontier_methods.py
"""
from __future__ import annotations

import sys
import pathlib
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "microsoft/phi-2"
N_CALIB = 2048
RFEAT = 128
K = 64
KLR = 8
AVG_KEEP = [0.7, 0.6]
LIMIT = 300
TASKS = ["boolq", "copa", "rte", "cb", "wic"]


def kmeans(X, k, iters=15):
    c = X[torch.randperm(X.shape[0], device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a


def mlp_fit(X, Y, hidden=128, steps=300, lr=5e-3, wd=1e-4):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            F.mse_loss(net(X), Y).backward()
            opt.step()
    return net.eval()


def ridge_fit(X, Y, lam=1e-2):
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device),
                              X.T @ Y)


def gm_forward(module, x):
    if module._gm_mode == "dense":
        return module._gm_orig(x)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    a = module.activation_fn(module.fc1(x2))
    grp, mean_a, gsz = module._gm_grp, module._gm_mean_a, module._gm_gsz
    Nn, dff = a.shape
    Gn = gsz.shape[0]
    z = (x2.float() - module._gm_xbar) @ module._gm_P
    score = module._gm_mlp(z)                              # predicted per-group residual
    order = score.argsort(dim=1, descending=True)         # high residual first = keep
    so = gsz[order]
    if module._gm_sel == "threshold":
        # drop low-residual tail until cumulative DROPPED predicted residual >= thresh
        sres = score.gather(1, order)                     # desc
        # cumulative residual of the DROPPED side (from the bottom)
        drop_cum = sres.flip(1).cumsum(1).flip(1)         # residual of this group + all below
        # a group is dropped if everything from it down sums under thresh... use tail sum
        # keep group g (in desc order) if cumulative-from-bottom up to it > thresh
        keep_ord = drop_cum > module._gm_thresh
        keep_ord[:, 0] = True                             # always keep top group
    else:
        budget = module._gm_keep * dff
        keep_ord = (so.cumsum(1) - so) < budget
    selg = torch.zeros(Nn, Gn, dtype=torch.bool, device=a.device)
    selg.scatter_(1, order, keep_ord)
    m = selg[:, grp].to(a.dtype)
    if module._gm_rep == "lr":
        rep = (mean_a.unsqueeze(0) + (z @ module._gm_Wlr).to(a.dtype))
    else:
        rep = mean_a.unsqueeze(0)
    return module.fc2(a * m + rep * (1 - m)).reshape(shape)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)
    layers = model.model.layers
    nL = len(layers)

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0]
    cap = {li: [] for li in range(nL)}
    hs = [layers[li].mlp.register_forward_pre_hook(
        (lambda li: (lambda _m, a: cap[li].append(
            a[0].detach().reshape(-1, a[0].shape[-1]))))(li)) for li in range(nL)]
    model(ids[:N_CALIB].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    HL = {li: torch.cat(cap[li]).float() for li in range(nL)}
    calib_scores = {}                                     # per-layer calib group scores

    for li in range(nL):
        mlp = layers[li].mlp
        mlp._gm_orig = mlp.forward
        mlp.forward = types.MethodType(gm_forward, mlp)
        mlp._gm_mode = "dense"
        Hc = HL[li]
        xbar = Hc.mean(0)
        _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
        P = Vt[:RFEAT].T
        a_c = mlp.activation_fn(mlp.fc1(Hc.half()).float())
        mean_a = a_c.mean(0)
        w1 = mlp.fc1.weight.float()
        grp = kmeans(w1 / (w1.norm(dim=1, keepdim=True) + 1e-8), K)
        idx = [(grp == g).nonzero().flatten() for g in range(K)]
        gsz = torch.tensor([len(ix) for ix in idx], device=dev, dtype=torch.float16)
        dev_a = a_c - mean_a
        fc2w = mlp.fc2.weight.float()
        gr = torch.stack([(dev_a[:, ix] @ fc2w[:, ix].T).norm(dim=1) for ix in idx], 1)
        z = (Hc - xbar) @ P
        mlp._gm_grp, mlp._gm_mean_a, mlp._gm_gsz = grp, mean_a.half(), gsz
        mlp._gm_xbar, mlp._gm_P = xbar, P
        mlp._gm_mlp = mlp_fit(z, gr)
        mlp._gm_Wlr = ridge_fit(z, a_c - mean_a)           # [RFEAT, dff] LR representative
        calib_scores[li] = mlp._gm_mlp(z).detach()         # [Tc,K]
        print(f"  built layer {li}", flush=True)

    # --- per-layer sensitivity: KL when ONLY layer L is MoEfied (uniform keep ref) ---
    @torch.no_grad()
    def dense_logits():
        return model(ids[:256].unsqueeze(0).to(dev)).logits[0].float().log_softmax(-1)

    def set_all(mode, keep=0.7, sel="budget", rep="static"):
        for li in range(nL):
            m = layers[li].mlp
            m._gm_mode, m._gm_keep, m._gm_sel, m._gm_rep = mode, keep, sel, rep

    set_all("dense")
    base = dense_logits()
    sens = torch.zeros(nL)
    for li in range(nL):
        layers[li].mlp._gm_mode = "moe"
        layers[li].mlp._gm_keep = 0.7
        layers[li].mlp._gm_sel = "budget"
        layers[li].mlp._gm_rep = "static"
        lg = model(ids[:256].unsqueeze(0).to(dev)).logits[0].float().log_softmax(-1)
        sens[li] = (base.exp() * (base - lg)).sum(-1).mean().item()
        layers[li].mlp._gm_mode = "dense"
        print(f"  sens layer {li}: {sens[li]:.4f}", flush=True)

    def alloc(avg, spread=0.25):
        s = (sens - sens.mean()) / (sens.std() + 1e-6)
        k = avg + spread * s
        for _ in range(20):
            k = k.clamp(0.35, 0.98)
            k = k + (avg - k.mean())
        return k.clamp(0.35, 0.98)

    def calib_thresh(avg):
        # per-layer threshold so avg kept-unit fraction == avg (per-token threshold sel)
        th = torch.zeros(nL)
        for li in range(nL):
            sc = calib_scores[li]                          # [Tc,K]
            gsz = layers[li].mlp._gm_gsz.float()
            order = sc.argsort(dim=1, descending=True)
            sres = sc.gather(1, order)
            so = gsz[order]
            drop_cum = sres.flip(1).cumsum(1).flip(1)
            lo, hi = 0.0, float(sres.sum(1).max())
            for _ in range(25):
                t = (lo + hi) / 2
                keep_ord = drop_cum > t
                keep_ord[:, 0] = True
                kept_units = (so * keep_ord).sum(1)
                frac = (kept_units / gsz.sum()).mean().item()
                if frac > avg:
                    lo = t
                else:
                    hi = t
            th[li] = (lo + hi) / 2
        return th

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)

    def run(tag):
        res = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, limit=LIMIT,
                              verbosity="ERROR", bootstrap_iters=0)
        r = res["results"]
        accs = {}
        for t in TASKS:
            d = r.get(t, {})
            met = next((k for k in d if k.startswith("acc")), None)
            accs[t] = d.get(met, float("nan")) if met else float("nan")
        vals = [v for v in accs.values() if v == v]
        avg = sum(vals) / len(vals)
        print(f"[{tag}] " + " ".join(f"{t}:{accs[t]:.3f}" for t in TASKS) +
              f" | AVG {avg:.4f}", flush=True)
        return avg

    print(f"\nPhi-2 frontier methods, K={K}, faithful lm-eval subset "
          f"{TASKS} (limit={LIMIT})\n")
    set_all("dense")
    run("dense")
    for avg in AVG_KEEP:
        kL = alloc(avg)
        thL = calib_thresh(avg)
        print(f"\n--- avg_keep={avg} (per-layer keep range "
              f"{kL.min():.2f}-{kL.max():.2f}) ---", flush=True)
        # uniform
        set_all("moe", keep=avg, sel="budget", rep="static")
        run(f"uniform k{avg}")
        # per-layer
        set_all("moe", sel="budget", rep="static")
        for li in range(nL):
            layers[li].mlp._gm_keep = float(kL[li])
        run(f"per-layer k{avg}")
        # per-token
        set_all("moe", sel="threshold", rep="static")
        for li in range(nL):
            layers[li].mlp._gm_thresh = float(thL[li])
        run(f"per-token k{avg}")
        # LR rep (uniform keep)
        set_all("moe", keep=avg, sel="budget", rep="lr")
        run(f"LR-rep k{avg}")
        # combined
        set_all("moe", sel="threshold", rep="lr")
        for li in range(nL):
            layers[li].mlp._gm_keep = float(kL[li])
            layers[li].mlp._gm_thresh = float(thL[li])
        run(f"combined k{avg}")

    print("\nIf any lever > uniform at the same avg_keep (esp. at 0.6) -> a real")
    print("compute-accuracy frontier push beyond G-MoEfication. If all ~ uniform ->")
    print("post-hoc compute reduction is intrinsically capped; certificate is the value.")


if __name__ == "__main__":
    main()
