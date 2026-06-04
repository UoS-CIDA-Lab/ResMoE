"""Experiment 117 — Reproduce G-MoEfication on Phi-2 (their benchmark) and test the
finer-granularity hypothesis on the SAME task: does larger K beat their K=64 at matched
85% FFN budget, zero-shot SuperGLUE?

Phi-2 is a dense GeLU model (32 layers, d=2560, d_ff=10240) -- exactly G-MoEfication's
Table-2 target. We MoEfy every FFN: group d_ff units into K experts (param-clustering on
fc1 rows), STATIC representative r_j = mean(gelu(h_j)) for dropped units (G-MoEfication),
self-supervised MLP router predicting per-group post-correction residual. Keep a fraction
of units per token. Compare zero-shot accuracy on a SuperGLUE subset (BoolQ, RTE, WiC, CB)
for: dense (100%) vs K in {64 (G-MoE granularity), 256 (finer, ours)} at keep {0.85,0.5}.

Self-contained eval (no lm-eval dep) via option log-likelihood; relative comparison under
a consistent harness is the point. Run: python3 experiments/117_phi2_gmoe_granularity.py
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
KEEP = [0.85, 0.5]
N_PER_TASK = 200


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
    a = module.activation_fn(module.fc1(x2))               # [N,dff]
    grp, mean_a, gsz = module._gm_grp, module._gm_mean_a, module._gm_gsz
    N, dff = a.shape
    Gn = gsz.shape[0]
    budget = module._gm_keep * dff
    z = (x2.float() - module._gm_xbar) @ module._gm_P
    score = module._gm_mlp(z)                              # [N,K]
    order = score.argsort(dim=1, descending=True)
    so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < budget
    selg = torch.zeros(N, Gn, dtype=torch.bool, device=a.device)
    selg.scatter_(1, order, keep_ord)
    m = selg[:, grp].to(a.dtype)
    kept = a * m + mean_a.unsqueeze(0) * (1 - m)
    return module.fc2(kept).reshape(shape)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)
    layers = model.model.layers
    nL = len(layers)

    # --- collect per-layer MLP inputs on calib text ---
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

    # --- patch all MLP forwards ---
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
            a_c = mlp.activation_fn(mlp.fc1(Hc.half()).float())   # [Tc,dff]
            mean_a = a_c.mean(0)
            w1 = mlp.fc1.weight.float()
            grp = kmeans(w1 / (w1.norm(dim=1, keepdim=True) + 1e-8), K)
            idx = [(grp == g).nonzero().flatten() for g in range(K)]
            gsz = torch.tensor([len(ix) for ix in idx], device=dev,
                               dtype=torch.float16)
            dev_a = a_c - mean_a
            fc2w = mlp.fc2.weight.float()
            gr = torch.stack([(dev_a[:, ix] @ fc2w[:, ix].T).norm(dim=1)
                              for ix in idx], 1)
            z = (Hc - xbar) @ P
            mlp._gm_grp = grp
            mlp._gm_mean_a = mean_a.half()
            mlp._gm_gsz = gsz
            mlp._gm_xbar = xbar
            mlp._gm_P = P
            mlp._gm_mlp = mlp_fit(z, gr)

    def set_cfg(mode, keep=0.85):
        for li in range(nL):
            layers[li].mlp._gm_mode = mode
            layers[li].mlp._gm_keep = keep

    # --- SuperGLUE-subset zero-shot eval (option log-likelihood) ---
    @torch.no_grad()
    def ll(prompt, cont):
        pi = tok(prompt, return_tensors="pt").input_ids.to(dev)
        ci = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        idsx = torch.cat([pi, ci], 1)
        lg = model(idsx).logits[0].float().log_softmax(-1)
        n = ci.shape[1]
        return lg[-n - 1:-1].gather(1, ci[0].unsqueeze(1)).sum().item()

    def acc_boolq(n):
        ds = load_dataset("super_glue", "boolq", split="validation",
                          trust_remote_code=True).select(range(n))
        c = 0
        for ex in ds:
            p = f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:"
            c += int((1 if ll(p, " yes") > ll(p, " no") else 0) == ex["label"])
        return c / len(ds)

    def acc_rte(n):
        ds = load_dataset("super_glue", "rte", split="validation",
                          trust_remote_code=True).select(range(n))
        c = 0
        for ex in ds:
            p = f"{ex['premise']}\nQuestion: {ex['hypothesis']} True or False?\nAnswer:"
            pred = 0 if ll(p, " True") > ll(p, " False") else 1
            c += int(pred == ex["label"])
        return c / len(ds)

    def acc_wic(n):
        ds = load_dataset("super_glue", "wic", split="validation",
                          trust_remote_code=True).select(range(n))
        c = 0
        for ex in ds:
            p = (f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\n"
                 f"Question: Is the word '{ex['word']}' used with the same meaning "
                 f"in both sentences?\nAnswer:")
            c += int((1 if ll(p, " yes") > ll(p, " no") else 0) == ex["label"])
        return c / len(ds)

    def acc_cb(n):
        ds = load_dataset("super_glue", "cb", split="validation",
                          trust_remote_code=True)
        ds = ds.select(range(min(n, len(ds))))
        c = 0
        for ex in ds:
            p = (f"{ex['premise']}\nQuestion: {ex['hypothesis']} true, false, "
                 f"or neither?\nAnswer:")
            sc = [ll(p, " true"), ll(p, " false"), ll(p, " neither")]
            c += int(int(torch.tensor(sc).argmax()) == ex["label"])
        return c / len(ds)

    def evaluate(tag):
        b = acc_boolq(min(N_PER_TASK, 150))
        r = acc_rte(N_PER_TASK)
        w = acc_wic(N_PER_TASK)
        cb = acc_cb(N_PER_TASK)
        avg = (b + r + w + cb) / 4
        print(f"  {tag:>16s} | BoolQ {b:.3f} RTE {r:.3f} WiC {w:.3f} CB {cb:.3f}"
              f" | AVG {avg:.3f}", flush=True)
        return avg

    print(f"\nPhi-2 zero-shot SuperGLUE-subset (N/task<={N_PER_TASK}); "
          f"all 32 FFNs G-MoEfied, STATIC rep, self-sup MLP router\n")
    set_cfg("dense")
    evaluate("dense (100%)")
    for K in KS:
        build_for_K(K)
        for keep in KEEP:
            set_cfg("moe", keep)
            evaluate(f"K={K} keep{int(keep*100)}%")

    print("\nIf K=256 (finer) > K=64 (G-MoEfication) at the same keep and approaches")
    print("dense -> finer granularity improves G-MoEfication on its own benchmark at")
    print("matched FFN budget. If K=256 ~ K=64 -> the finer-granularity gain seen on")
    print("OLMoE/FFN-L2 does not transfer to Phi-2 task accuracy.")


if __name__ == "__main__":
    main()
