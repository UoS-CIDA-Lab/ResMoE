"""Experiment 134 — Can STATIC analysis of expert interactions give a smarter selection
than independent top-k (greedy)? Tests the user's idea: use a precomputed (token-
independent) cross-correlation structure C-bar between groups to do interaction-aware
selection, rather than ranking groups independently.

Isolate SELECTION quality using ORACLE true per-group magnitudes (no router prediction).
Three rules, all evaluated by the TRUE per-token error ||sum r_dropped||:
  greedy      : top-k by per-token ||r_g(t)||           (independent)
  static-int  : local-search minimizing a SURROGATE error  E(D)=sum_{i,j in D} ||r_i(t)|| ||r_j(t)|| C-bar[i,j]
                where C-bar = mean_t normalized Gram (STATIC off-diagonal, per-token diagonal)
  pt-optimal  : local-search on the TRUE per-token Gram   (ceiling, exp 133)
If static-int ~ greedy << pt-optimal => interaction is per-token, static can't capture it.
If static-int ~ pt-optimal => static interaction structure DOES enable smarter selection.
Also report mean |C-bar off-diagonal| (how non-orthogonal the static structure is).
Run: python3 experiments/134_static_interaction_select.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 2048
KEEPS = [0.25, 0.5]
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


def quad(m, Gram):
    return torch.einsum('ti,tij,tj->t', m, Gram, m).clamp_min(0)


def local_search(m, Gram, iters=40):
    T, K = m.shape
    diag = torch.diagonal(Gram, dim1=1, dim2=2)
    for _ in range(iters):
        s = torch.einsum('tij,tj->ti', Gram, m)
        rem = (-2 * s + diag)
        add = (2 * s + diag)
        delta = rem.unsqueeze(2) + add.unsqueeze(1) - 2 * Gram
        invalid = (m.unsqueeze(2) < 0.5) | (m.unsqueeze(1) > 0.5)
        delta = delta.masked_fill(invalid, float('inf'))
        flat = delta.reshape(T, -1)
        best, arg = flat.min(1)
        i = arg // K; j = arg % K
        improve = best < -1e-9
        if not improve.any():
            break
        ti = improve.nonzero().flatten()
        m[ti, i[ti]] = 0.0
        m[ti, j[ti]] = 1.0
    return m


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind, K in MODELS:
        print(f"\n{'='*64}\n{name} (K={K})\n{'='*64}", flush=True)
        try:
          tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
          ids = tok(text, return_tensors="pt").input_ids[0][:N_TOK]
          Loader = AutoModel if kind == "mlm" else AutoModelForCausalLM
          model = Loader.from_pretrained(name, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
          if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
          torch.set_grad_enabled(False)
          layers = layer_list(tag, model); nL = len(layers); li = nL // 2
          up, dn = linears(tag, layers[li])
          capx, capa = [], []
          h1 = up.register_forward_pre_hook(lambda _m, a: capx.append(a[0].reshape(-1, a[0].shape[-1]).float()))
          h2 = dn.register_forward_pre_hook(lambda _m, a: capa.append(a[0].reshape(-1, a[0].shape[-1]).float()))
          chunk = 512 if tag == "bert" else 1024
          for c0 in range(0, ids.shape[0], chunk):
            model(ids[c0:c0 + chunk].unsqueeze(0).to(dev))
          h1.remove(); h2.remove()
          a = torch.cat(capa).to(dev); dff = a.shape[1]
          Wd = dn.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
          Wup = up.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
          abar = a.mean(0); dev_a = a - abar
          y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6
          T = a.shape[0]
          grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
          idxs = [(grp == g).nonzero().flatten() for g in range(K)]
          Gram = torch.zeros(T, K, K, device=dev)
          for t0 in range(0, T, 256):
            R = torch.stack([dev_a[t0:t0+256, ix] @ Wd[:, ix].T for ix in idxs], 1)
            Gram[t0:t0+256] = torch.einsum('cgd,chd->cgh', R, R)
          rnorm = torch.diagonal(Gram, dim1=1, dim2=2).clamp_min(1e-12).sqrt()       # [T,K]
          # static normalized correlation C-bar = mean_t Gram_t / (||r_i|| ||r_j||)
          Cnorm = Gram / (rnorm.unsqueeze(2) * rnorm.unsqueeze(1))
          Cbar = Cnorm.mean(0)                                                       # [K,K]
          offmag = (Cbar - torch.diag(torch.diagonal(Cbar))).abs().mean().item()
          # surrogate per-token Gram: per-token magnitudes, STATIC correlation
          Gsurr = (rnorm.unsqueeze(2) * rnorm.unsqueeze(1)) * Cbar.unsqueeze(0)      # [T,K,K]
          print(f"  layer {li}, dff={dff}, mean|C-bar off-diag| = {offmag:.4f}", flush=True)
          print(f"    {'keep':>5s} | {'greedy':>7s} | {'static-int':>10s} | {'pt-optimal':>10s}")
          for keep in KEEPS:
            drop = K - int(round(keep * K))
            gd = (rnorm * rnorm).argsort(1)[:, :drop]
            gm = torch.zeros(T, K, device=dev); gm.scatter_(1, gd, 1.0)
            e_greedy = (quad(gm, Gram).sqrt() / ynorm).mean().item()
            m_si = local_search(gm.clone(), Gsurr)            # select on surrogate
            e_si = (quad(m_si, Gram).sqrt() / ynorm).mean().item()   # eval on TRUE
            m_opt = local_search(gm.clone(), Gram)
            e_opt = (quad(m_opt, Gram).sqrt() / ynorm).mean().item()
            print(f"    {int(keep*100):>4d}% | {e_greedy*100:6.1f}% | {e_si*100:9.1f}% | {e_opt*100:9.1f}%", flush=True)
          del model
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: static-int ~ greedy << pt-optimal => interaction is per-token; static analysis")
    print("can't capture it. static-int ~ pt-optimal => static interaction enables smarter selection.")


if __name__ == "__main__":
    main()
