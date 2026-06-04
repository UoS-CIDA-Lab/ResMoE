"""Experiment 135 — Does the greedy->optimal selection gap GROW with finer K (more experts)?
exp 133 saw the gap grow G16->K64 (0.9->1.9pt @keep50 mBERT). User: push K higher.
Sweep K in {64,128,256,512}, measure greedy vs local-search-optimal (token-chunked for
memory; validated == brute at G=16 in exp 133) at keep 25/50, plus ortho-ratio.
NOTE: any gap is still ORACLE-level (needs full per-token Gram = full FFN) — this measures
whether more interaction-headroom EXISTS at finer K, not whether it's deployably capturable.
Run: python3 experiments/135_K_sweep_gap.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 2048
KS = [64, 128, 256, 512]
KEEPS = [0.25, 0.5]
MODELS = [
    ("bert",       "bert-base-multilingual-cased",  "mlm",    None),
    ("santacoder", "bigcode/gpt_bigcode-santacoder", "causal", None),
    ("falcon",     "tiiuae/falcon-7b",              "causal", None),
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


def local_search_chunk(m, Gram, iters=60):
    T, K = m.shape
    diag = torch.diagonal(Gram, dim1=1, dim2=2)
    for _ in range(iters):
        s = torch.einsum('tij,tj->ti', Gram, m)
        delta = (-2 * s + diag).unsqueeze(2) + (2 * s + diag).unsqueeze(1) - 2 * Gram
        invalid = (m.unsqueeze(2) < 0.5) | (m.unsqueeze(1) > 0.5)
        delta = delta.masked_fill(invalid, float('inf'))
        best, arg = delta.reshape(T, -1).min(1)
        i = arg // K; j = arg % K
        improve = best < -1e-9
        if not improve.any():
            break
        ti = improve.nonzero().flatten()
        m[ti, i[ti]] = 0.0; m[ti, j[ti]] = 1.0
    return m


def local_search(gm, Gram, tchunk=256):
    out = torch.empty_like(gm)
    for t0 in range(0, gm.shape[0], tchunk):
        out[t0:t0+tchunk] = local_search_chunk(gm[t0:t0+tchunk].clone(), Gram[t0:t0+tchunk])
    return out


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind, _ in MODELS:
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
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
          del model; gc.collect(); torch.cuda.empty_cache()
          print(f"  layer {li}, dff={dff}")
          print(f"  {'K':>5s} | {'grp':>4s} | {'keep':>5s} | {'greedy':>7s} | {'optimal':>7s} | {'gap':>5s} | ortho")
          print("  " + "-" * 64)
          for K in KS:
            if K > dff:
                continue
            grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
            idxs = [(grp == g).nonzero().flatten() for g in range(K)]
            Gram = torch.zeros(T, K, K, device=dev)
            for t0 in range(0, T, 128):
                R = torch.stack([dev_a[t0:t0+128, ix] @ Wd[:, ix].T for ix in idxs], 1)
                Gram[t0:t0+128] = torch.einsum('cgd,chd->cgh', R, R)
            rn2 = torch.diagonal(Gram, dim1=1, dim2=2).clamp_min(0)
            for keep in KEEPS:
                drop = K - int(round(keep * K))
                gd = rn2.argsort(1)[:, :drop]
                gm = torch.zeros(T, K, device=dev); gm.scatter_(1, gd, 1.0)
                eg = quad(gm, Gram)
                ratio = (eg / (rn2 * gm).sum(1).clamp_min(1e-9)).mean().item()
                lm = local_search(gm, Gram)
                el = quad(lm, Gram)
                g = (eg.sqrt() / ynorm).mean().item() * 100
                o = (el.sqrt() / ynorm).mean().item() * 100
                print(f"  {K:>5d} | {dff//K:>4d} | {int(keep*100):>4d}% | {g:6.1f}% | {o:6.1f}% | "
                      f"{g-o:4.1f}pt | {ratio:.3f}", flush=True)
            del Gram; gc.collect(); torch.cuda.empty_cache()
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: does (greedy-optimal) gap grow with K? still oracle-level (full Gram=full FFN).")


if __name__ == "__main__":
    main()
