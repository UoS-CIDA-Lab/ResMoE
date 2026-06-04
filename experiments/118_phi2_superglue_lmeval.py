"""Experiment 118 — FAITHFUL SuperGLUE (lm-eval-harness) for G-MoEfied Phi-2: does finer
granularity (K=256) beat G-MoEfication's K=64 at matched 85% FFN budget on THEIR exact
benchmark/metrics?

This is the rigorous version of exp 117: real lm-eval SuperGLUE (8 tasks: boolq, cb,
copa, multirc, record, rte, wic, wsc) instead of a custom 4-task subset. Phi-2 dense GeLU
FFNs are G-MoEfied (STATIC representative, self-sup MLP router), wrapped as HFLM, run
zero-shot. Compare dense vs K=64 vs K=256 at keep 0.85. (limit per task for tractability;
relative comparison under the identical harness/metrics is the point.)

Run: python3 experiments/118_phi2_superglue_lmeval.py
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
KS = [64, 256]
KEEP = 0.85
LIMIT = 200
TASKS = ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]


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


def gm_forward(module, x):
    if module._gm_mode == "dense":
        return module._gm_orig(x)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    a = module.activation_fn(module.fc1(x2))
    grp, mean_a, gsz = module._gm_grp, module._gm_mean_a, module._gm_gsz
    N, dff = a.shape
    Gn = gsz.shape[0]
    budget = module._gm_keep * dff
    z = (x2.float() - module._gm_xbar) @ module._gm_P
    score = module._gm_mlp(z)
    order = score.argsort(dim=1, descending=True)
    so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < budget
    selg = torch.zeros(N, Gn, dtype=torch.bool, device=a.device)
    selg.scatter_(1, order, keep_ord)
    m = selg[:, grp].to(a.dtype)
    return module.fc2(a * m + mean_a.unsqueeze(0) * (1 - m)).reshape(shape)


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

    for li in range(nL):
        mlp = layers[li].mlp
        mlp._gm_orig = mlp.forward
        mlp.forward = types.MethodType(gm_forward, mlp)
        mlp._gm_mode = "dense"

    def build_for_K(K):
        print(f"  building K={K} ...", flush=True)
        for li in range(nL):
            mlp = layers[li].mlp
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
            gr = torch.stack([(dev_a[:, ix] @ fc2w[:, ix].T).norm(dim=1)
                              for ix in idx], 1)
            z = (Hc - xbar) @ P
            mlp._gm_grp, mlp._gm_mean_a, mlp._gm_gsz = grp, mean_a.half(), gsz
            mlp._gm_xbar, mlp._gm_P, mlp._gm_mlp = xbar, P, mlp_fit(z, gr)

    def set_cfg(mode, keep=KEEP):
        for li in range(nL):
            layers[li].mlp._gm_mode = mode
            layers[li].mlp._gm_keep = keep

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)

    def run(tag):
        res = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0,
                              limit=LIMIT, verbosity="ERROR", bootstrap_iters=0)
        r = res["results"]
        accs = {}
        for t in TASKS:
            d = r.get(t, {})
            met = next((k for k in d if k.startswith("acc")), None)
            accs[t] = d.get(met, float("nan")) if met else float("nan")
        avg = sum(v for v in accs.values() if v == v) / len(accs)
        cells = " ".join(f"{t}:{accs[t]:.3f}" for t in TASKS)
        print(f"\n[{tag}] {cells}\n[{tag}] SuperGLUE-avg(acc) = {avg:.4f}", flush=True)
        return avg

    print(f"\nPhi-2 FAITHFUL SuperGLUE (lm-eval, limit={LIMIT}/task), keep={KEEP}\n")
    set_cfg("dense")
    run("dense")
    for K in KS:
        build_for_K(K)
        set_cfg("moe", KEEP)
        run(f"K={K}")

    print("\nK=256 > K=64 toward dense on lm-eval SuperGLUE -> finer granularity improves")
    print("G-MoEfication on its exact benchmark at matched 85% FFN budget.")


if __name__ == "__main__":
    main()
