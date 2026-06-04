"""Experiment 138 — EXPERT CONSTRUCTION for ROUTABILITY (user pivot).
Selection given fixed experts is exhausted (greedy near-optimal; deployable->oracle gap ~12pt
is the binding wall, exp137). But the GROUPING (how we build experts) we held ~fixed.
Key asymmetry: oracle drop-error is partition-INVARIANT (exp114), but DEPLOYABLE error can
depend on grouping. So: can we group neurons so a router PREDICTS group-need better from x,
shrinking the deployable->oracle gap WITHOUT changing the floor?

Groupings (K=64): random | wup-cos (input dir, default) | vout-cos (output dir) |
coact (cluster neurons by activation profile across tokens -> coherent, x-predictable groups).
For each: oracle error (true ||r_g|| top-k) vs deployable error (reg MLP from x). Report the gap.
Smaller deployable gap for some grouping => construct-for-routability is a real lever.
Run: python3 experiments/138_construct_for_routability.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 2048
K = 64
KEEPS = [0.25, 0.5]
SPLIT = 1536
MODELS = [
    ("bert",       "bert-base-multilingual-cased",  "mlm"),
    ("santacoder", "bigcode/gpt_bigcode-santacoder", "causal"),
    ("falcon",     "tiiuae/falcon-7b",              "causal"),
]


def linears(tag, layer):
    if tag == "bert":
        return layer.intermediate.dense, layer.output.dense
    if tag == "santacoder":
        return layer.mlp.c_fc, layer.mlp.c_proj
    return layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h


def layer_list(tag, model):
    return model.encoder.layer if tag == "bert" else model.transformer.h


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


def mlp_fit(X, Y, steps=500, lr=4e-3, hidden=256):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); F.mse_loss(net(X), Y).backward(); opt.step()
    return net.eval()


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind in MODELS:
        print(f"\n{'='*72}\n{name}\n{'='*72}", flush=True)
        try:
          tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
          ids = tok(text, return_tensors="pt").input_ids[0][:N_TOK]
          Loader = AutoModel if kind == "mlm" else AutoModelForCausalLM
          model = Loader.from_pretrained(name, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
          if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
          torch.set_grad_enabled(False)
          layers = layer_list(tag, model); li = len(layers) // 2
          up, dn = linears(tag, layers[li])
          capx, capa = [], []
          h1 = up.register_forward_pre_hook(lambda _m, a: capx.append(a[0].reshape(-1, a[0].shape[-1]).float()))
          h2 = dn.register_forward_pre_hook(lambda _m, a: capa.append(a[0].reshape(-1, a[0].shape[-1]).float()))
          ch = 512 if tag == "bert" else 1024
          for c0 in range(0, ids.shape[0], ch):
            model(ids[c0:c0 + ch].unsqueeze(0).to(dev))
          h1.remove(); h2.remove()
          a = torch.cat(capa).to(dev); x = torch.cat(capx).to(dev); dff = a.shape[1]
          Wd = dn.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
          Wup = up.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
          abar = a.mean(0); dev_a = a - abar
          y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6
          T = a.shape[0]
          xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
          Z = ((x - xbar) @ Vt[:128].T).float()
          del model; gc.collect(); torch.cuda.empty_cache()
          tr, ev = slice(0, SPLIT), slice(SPLIT, T)
          # candidate groupings
          rand = torch.randint(0, K, (dff,), device=dev,
                               generator=torch.Generator(device=dev).manual_seed(0))
          groupings = {
              "random":   rand,
              "wup-cos":  kmeans(F.normalize(Wup, dim=1), K, seed=0),
              "vout-cos": kmeans(F.normalize(Wd.T, dim=1), K, seed=0),
              "coact":    kmeans(F.normalize((dev_a[:SPLIT].T), dim=1), K, seed=0),  # neuron activation profiles
          }
          print(f"  layer {li}, dff={dff}, K={K}")
          print(f"  {'grouping':>9s} | {'keep':>5s} | {'oracle':>7s} | {'deployable':>10s} | {'gap':>5s}")
          print("  " + "-" * 52)
          for gname, grp in groupings.items():
            idxs = [(grp == g).nonzero().flatten() for g in range(K)]
            # per-token group residual magnitude ||r_g(t)|| (greedy/oracle scores by norm)
            rn = torch.zeros(T, K, device=dev)
            for g in range(K):
                ix = idxs[g]
                if ix.numel() == 0:
                    continue
                rg = dev_a[:, ix] @ Wd[:, ix].T           # [T,d]
                rn[:, g] = rg.norm(dim=1)
            for keep in KEEPS:
                drop = K - int(round(keep * K))
                # oracle: drop smallest ||r_g||; error = ||sum dropped r_g||
                gd = rn.argsort(1)[:, :drop]
                # need true error: recompute dropped-sum norm per token
                def err_from_keepscore(score, sl):
                    kk = K - drop
                    tk = score.argsort(1, descending=True)[:, :kk]
                    keepm = torch.zeros(score.shape[0], K, device=dev); keepm.scatter_(1, tk, 1.0)
                    # reconstruct error = || sum_{dropped} r_g || ; build per token
                    err = torch.zeros(score.shape[0], Wd.shape[0], device=dev)
                    sub = dev_a[sl]
                    for g in range(K):
                        ix = idxs[g]
                        if ix.numel() == 0:
                            continue
                        drop_g = (1 - keepm[:, g]).unsqueeze(1)
                        err += drop_g * (sub[:, ix] @ Wd[:, ix].T)
                    return (err.norm(dim=1) / ynorm[sl]).mean().item() * 100
                e_oracle = err_from_keepscore(rn[ev], ev)
                reg = mlp_fit(Z[tr], rn[tr])
                e_dep = err_from_keepscore(reg(Z[ev]), ev)
                print(f"  {gname:>9s} | {int(keep*100):>4d}% | {e_oracle:6.1f}% | {e_dep:9.1f}% | "
                      f"{e_dep-e_oracle:4.1f}pt", flush=True)
            del rn; gc.collect(); torch.cuda.empty_cache()
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: does any grouping shrink the deployable->oracle GAP (vs random/wup-cos)?")
    print("oracle ~equal across groupings (partition-invariant, exp114). If a grouping gives a")
    print("smaller deployable gap => constructing experts for ROUTABILITY is a real lever.")


if __name__ == "__main__":
    main()
