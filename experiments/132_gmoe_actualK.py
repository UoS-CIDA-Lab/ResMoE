"""Experiment 132 — Re-measure FFN-output error at G-MoEfication's ACTUAL expert count
K (paper line 735: K=64, except Falcon-7B K=128) for apples-to-apples with the paper.

Reconciles "why does mBERT barely drop at ~50% in the paper" vs our G=16 46.8%:
quantify the oracle/greedy FFN error at THEIR granularity (finer than our G=16) + the
deployable (trained-router) error + the per-UNIT floor. brute-optimal is infeasible at
K=64 (C(64,32)) so we report oracle(=true ||r_g|| top-k) and deployable reg-MLP only.

Models (their own GeLU baseline): mBERT(K64), SantaCoder-1.1B(K64), Falcon-7B(K128).
Gram accumulated in token-chunks to avoid OOM on Falcon's wide FFN.
Run: python3 experiments/132_gmoe_actualK.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 2048
KEEPS = [0.25, 0.5, 0.85]
SPLIT = 1536
MODELS = [
    ("bert",       "bert-base-multilingual-cased",  "mlm",    64),
    ("santacoder", "bigcode/gpt_bigcode-santacoder", "causal", 64),
    ("falcon",     "tiiuae/falcon-7b",              "causal", 128),
]


def linears(tag, layer):
    if tag == "bert":
        return layer.intermediate.dense, layer.output.dense
    if tag == "santacoder":
        return layer.mlp.c_fc, layer.mlp.c_proj
    if tag == "falcon":
        return layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h


def layer_list(tag, model):
    return model.encoder.layer if tag == "bert" else model.transformer.h


def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a


def mlp_fit(X, Y, steps=400, lr=5e-3, hidden=128):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); F.mse_loss(net(X), Y).backward(); opt.step()
    return net.eval()


def analyze(a, x, Wup, Wd, dev, K):
    T, dff = a.shape
    abar = a.mean(0); dev_a = a - abar
    vnorm = Wd.norm(dim=0)
    y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6
    out = {}
    # per-UNIT oracle floor (individual neurons)
    score_u = dev_a.abs() * vnorm
    for keep in KEEPS:
        k = int(round(keep * dff))
        idx = score_u.argsort(1, descending=True)
        mask = torch.zeros_like(a); mask.scatter_(1, idx[:, :k], 1.0)
        out[f"unit{keep}"] = (((y - (a * mask + abar * (1 - mask)) @ Wd.T).norm(dim=1)) / ynorm).mean().item()
    # K groups
    grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
    idxs = [(grp == g).nonzero().flatten() for g in range(K)]
    # Gram[T,K,K] + rnorm[T,K], token-chunked to bound memory
    Gram = torch.zeros(T, K, K, device=dev)
    CH = 256
    for t0 in range(0, T, CH):
        R = torch.stack([dev_a[t0:t0+CH, ix] @ Wd[:, ix].T for ix in idxs], 1)  # [c,K,d]
        Gram[t0:t0+CH] = torch.einsum('cgd,chd->cgh', R, R)
    rnorm = torch.diagonal(Gram, dim1=1, dim2=2).clamp_min(0).sqrt()
    xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
    Z = ((x - xbar) @ Vt[:128].T).float()
    tr, ev = slice(0, SPLIT), slice(SPLIT, T)

    def kerr(score, keep, sl):
        k = int(round(keep * K)); tk = score.argsort(1, descending=True)[:, :k]
        km = torch.zeros(score.shape[0], K, device=dev); km.scatter_(1, tk, 1.0); dm = 1 - km
        return (torch.einsum('ti,tij,tj->t', dm, Gram[sl], dm).clamp_min(0).sqrt() / ynorm[sl]).mean().item()
    for keep in KEEPS:
        reg = mlp_fit(Z[tr], rnorm[tr])
        out[f"oracle{keep}"] = kerr(rnorm, keep, slice(0, T))      # = greedy (true ||r_g|| top-k)
        out[f"reg{keep}"] = kerr(reg(Z[ev]), keep, ev)
    return out


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind, K in MODELS:
        print(f"\n{'='*66}\n{name} (K={K})\n{'='*66}", flush=True)
        try:
          tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
          ids = tok(text, return_tensors="pt").input_ids[0][:N_TOK]
          Loader = AutoModel if kind == "mlm" else AutoModelForCausalLM
          model = Loader.from_pretrained(name, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
          if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
          torch.set_grad_enabled(False)
          layers = layer_list(tag, model); nL = len(layers)
          targets = sorted({nL // 4, nL // 2, 3 * nL // 4})
          capx, capa = {li: [] for li in targets}, {li: [] for li in targets}
          hs = []
          for li in targets:
            up, dn = linears(tag, layers[li])
            hs.append(up.register_forward_pre_hook((lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
            hs.append(dn.register_forward_pre_hook((lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
          chunk = 512 if tag == "bert" else 1024
          for c0 in range(0, ids.shape[0], chunk):
            model(ids[c0:c0 + chunk].unsqueeze(0).to(dev))
          for h in hs:
            h.remove()
          rows = []
          for li in targets:
            up, dn = linears(tag, layers[li])
            a = torch.cat(capa[li]); x = torch.cat(capx[li]); dff = a.shape[1]
            Wup = up.weight.detach().float();  Wup = Wup if Wup.shape[0] == dff else Wup.T
            Wd = dn.weight.detach().float();   Wd = Wd if Wd.shape[1] == dff else Wd.T
            r = analyze(a.to(dev), x.to(dev), Wup.to(dev), Wd.to(dev), dev, K); r["dff"] = dff
            rows.append(r); print(f"    layer {li} (dff={dff}) done", flush=True)
          m = lambda k: float(np.mean([r[k] for r in rows]))
          print(f"\n  [{name}] K={K}, dff={rows[0]['dff']} (group size {rows[0]['dff']//K}), mean/{len(rows)} layers")
          print(f"    {'keep':>5s} | {'oracle@K':>8s} | {'reg-MLP@K':>9s} | {'per-unit oracle':>15s}")
          for keep in KEEPS:
            print(f"    {int(keep*100):>4d}% | {m(f'oracle{keep}')*100:7.1f}% | {m(f'reg{keep}')*100:8.1f}% | "
                  f"{m(f'unit{keep}')*100:14.1f}%", flush=True)
          del model
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nDONE. oracle@K = best selection at the paper's K (still fidelity, not task).")
    print("Compare to G=16 (exp131) and per-unit floor: finer K lowers FFN error, but it")
    print("stays substantial => task survival is residual-absorption + rep-value + finetune, not low FFN error.")


if __name__ == "__main__":
    main()
