"""Experiment 122 — The constructive answer: PRESERVE-and-quantize beats REMOVE, on the
compute-accuracy frontier of Phi-2 FFNs (faithful lm-eval SuperGLUE).

Every removal-based lever (8 of them) failed because computation lives in fine structure
(cancelling sums, orthogonal residuals, low-energy spectral tail) that removal destroys.
Quantization is the one thing that works (exp 100/101: quant >> prune) because it PRESERVES
all computation and only lowers precision. Here we test that as the practical compute
lever: compare at MATCHED effective compute (vs dense fp16):
  MoEfication-drop keep k  -> eff = k         (fp16, k% of units)
  quantize full b-bit      -> eff = b/16      (100% units, b-bit)
  hybrid keep k + b-bit    -> eff = k*b/16
If quant/hybrid Pareto-dominate drop -> for gated/dense FFNs the right compute reduction
is precision (preserve computation), not structural removal -- a concrete, impactful,
mechanism-grounded prescription. (Accuracy via simulated quantization; compute model is
first-order bits*ops; quant also dominates the memory/bandwidth axis that bounds LLM
inference.)

Run: python3 experiments/122_phi2_quant_vs_moe.py
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
LIMIT = 200
TASKS = ["boolq", "copa", "rte", "cb", "wic"]
# (name, mode, keep, bits, eff-compute vs dense)
CONFIGS = [
    ("dense",            "dense", 1.0, 16, 1.00),
    ("moe k0.5",         "moe",   0.5, 16, 0.50),
    ("quant 8b",         "quant", 1.0, 8,  0.50),
    ("moe k0.25",        "moe",   0.25, 16, 0.25),
    ("quant 4b",         "quant", 1.0, 4,  0.25),
    ("hybrid k0.5+8b",   "hybrid", 0.5, 8,  0.25),
    ("quant 3b",         "quant", 1.0, 3,  0.19),
]


def quant(W, b):
    if b >= 16:
        return W
    qmax = 2 ** (b - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return (torch.round(W / s).clamp(-qmax - 1, qmax) * s)


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
    useq = module._gm_mode in ("quant", "hybrid")
    W1 = module._gm_W1q if useq else module.fc1.weight
    W3 = module._gm_W3q if useq else module.fc2.weight
    b1, b3 = module.fc1.bias, module.fc2.bias
    a = module.activation_fn(x2 @ W1.T + (b1 if b1 is not None else 0.0))
    if module._gm_mode in ("moe", "hybrid"):
        grp, mean_a, gsz = module._gm_grp, module._gm_mean_a, module._gm_gsz
        Nn, dff = a.shape
        Gn = gsz.shape[0]
        budget = module._gm_keep * dff
        z = (x2.float() - module._gm_xbar) @ module._gm_P
        score = module._gm_mlp(z)
        order = score.argsort(dim=1, descending=True)
        so = gsz[order]
        keep_ord = (so.cumsum(1) - so) < budget
        selg = torch.zeros(Nn, Gn, dtype=torch.bool, device=a.device)
        selg.scatter_(1, order, keep_ord)
        m = selg[:, grp].to(a.dtype)
        a = a * m + mean_a.unsqueeze(0) * (1 - m)
    out = a @ W3.T + (b3 if b3 is not None else 0.0)
    return out.reshape(shape)


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

    print("building MoEfication params + quant weights ...", flush=True)
    for li in range(nL):
        mlp = layers[li].mlp
        mlp._gm_orig = mlp.forward
        mlp.forward = types.MethodType(gm_forward, mlp)
        mlp._gm_mode = "dense"
        Hc = torch.cat(cap[li]).float()
        xbar = Hc.mean(0)
        _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
        P = Vt[:RFEAT].T
        a_c = mlp.activation_fn(mlp.fc1(Hc.half()).float())
        mean_a = a_c.mean(0)
        w1 = mlp.fc1.weight.float()
        grp = kmeans(w1 / (w1.norm(dim=1, keepdim=True) + 1e-8), K)
        idx = [(grp == g).nonzero().flatten() for g in range(K)]
        gsz = torch.tensor([len(ix) for ix in idx], device=dev, dtype=torch.float16)
        fc2w = mlp.fc2.weight.float()
        gr = torch.stack([((a_c - mean_a)[:, ix] @ fc2w[:, ix].T).norm(dim=1)
                          for ix in idx], 1)
        z = (Hc - xbar) @ P
        mlp._gm_grp, mlp._gm_mean_a, mlp._gm_gsz = grp, mean_a.half(), gsz
        mlp._gm_xbar, mlp._gm_P, mlp._gm_mlp = xbar, P, mlp_fit(z, gr)

    def set_cfg(mode, keep, bits):
        for li in range(nL):
            mlp = layers[li].mlp
            mlp._gm_mode, mlp._gm_keep = mode, keep
            if mode in ("quant", "hybrid"):
                mlp._gm_W1q = quant(mlp.fc1.weight.float(), bits).half()
                mlp._gm_W3q = quant(mlp.fc2.weight.float(), bits).half()

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)

    def run(tag, eff):
        res = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, limit=LIMIT,
                              verbosity="ERROR", bootstrap_iters=0)
        r = res["results"]
        accs = {t: next((r[t][k] for k in r.get(t, {}) if k.startswith("acc")),
                        float("nan")) for t in TASKS}
        vals = [v for v in accs.values() if v == v]
        avg = sum(vals) / len(vals)
        print(f"[{tag:>16s} | eff {eff:.2f}] " +
              " ".join(f"{t}:{accs[t]:.3f}" for t in TASKS) + f" | AVG {avg:.4f}",
              flush=True)
        return avg

    print(f"\nPhi-2 quant(preserve) vs MoE(remove) frontier, faithful lm-eval {TASKS}\n")
    for name, mode, keep, bits, eff in CONFIGS:
        set_cfg(mode, keep, bits)
        run(name, eff)

    print("\nCompare at matched eff-compute: {0.50: moe-k0.5 vs quant-8b}, "
          "{0.25: moe-k0.25 vs quant-4b vs hybrid}. If quant/hybrid >> moe -> for")
    print("gated/dense FFNs, reduce PRECISION (preserve computation), don't REMOVE it.")


if __name__ == "__main__":
    main()
