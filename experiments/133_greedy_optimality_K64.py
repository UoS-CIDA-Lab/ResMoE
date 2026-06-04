"""Experiment 133 — Is greedy really (near-)optimal at G-MoE's ACTUAL K=64/128?
exp 130/131 brute-forced optimal only at G=16. Here verify at the real K via:
 (a) orthogonality ratio of the greedy drop-set: ||sum r_dropped||^2 / sum ||r_g||^2.
     ==1 => residuals orthogonal => greedy PROVABLY optimal (no cross-cancellation).
 (b) LOCAL SEARCH (iterated best single kept<->dropped swap on the binary quadratic
     E(m)=m^T Gram m) starting from greedy: how much can it beat greedy?
 Validation: at G=16 also brute-force; local-search should reach the brute optimum.
Models: mBERT(K64), SantaCoder(K64), Falcon-7B(K128). keep 25/50.
Run: python3 experiments/133_greedy_optimality_K64.py
"""
from __future__ import annotations
import sys, pathlib, itertools, gc
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
    """Iterated best-improving single swap (drop i in, kept j out). m: [T,K] drop mask."""
    T, K = m.shape
    diag = torch.diagonal(Gram, dim1=1, dim2=2)              # [T,K]
    for _ in range(iters):
        s = torch.einsum('tij,tj->ti', Gram, m)              # [T,K]
        # swap drop->keep group i (m_i=1) with keep->drop group j (m_j=0)
        # delta = (-2 s_i + G_ii) + (2 s_j - 2 G_ij + G_jj)
        rem = (-2 * s + diag)                                 # [T,K] effect of removing i (i currently dropped)
        add = (2 * s + diag)                                  # base add effect for j (before -2G_ij)
        delta = rem.unsqueeze(2) + add.unsqueeze(1) - 2 * Gram  # [T,K,K]  [t,i,j]
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
          layers = layer_list(tag, model); nL = len(layers)
          li = nL // 2
          up, dn = linears(tag, layers[li])
          capx, capa = [], []
          h1 = up.register_forward_pre_hook(lambda _m, a: capx.append(a[0].reshape(-1, a[0].shape[-1]).float()))
          h2 = dn.register_forward_pre_hook(lambda _m, a: capa.append(a[0].reshape(-1, a[0].shape[-1]).float()))
          chunk = 512 if tag == "bert" else 1024
          for c0 in range(0, ids.shape[0], chunk):
            model(ids[c0:c0 + chunk].unsqueeze(0).to(dev))
          h1.remove(); h2.remove()
          a = torch.cat(capa).to(dev); x = torch.cat(capx).to(dev); dff = a.shape[1]
          Wd = dn.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
          Wup = up.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
          abar = a.mean(0); dev_a = a - abar
          y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6
          T = a.shape[0]

          def run_at(Kc, brute=False):
            grp = kmeans(F.normalize(Wup, dim=1), Kc, seed=0)
            idxs = [(grp == g).nonzero().flatten() for g in range(Kc)]
            Gram = torch.zeros(T, Kc, Kc, device=dev)
            for t0 in range(0, T, 256):
                R = torch.stack([dev_a[t0:t0+256, ix] @ Wd[:, ix].T for ix in idxs], 1)
                Gram[t0:t0+256] = torch.einsum('cgd,chd->cgh', R, R)
            rnorm2 = torch.diagonal(Gram, dim1=1, dim2=2).clamp_min(0)
            out = {}
            for keep in KEEPS:
                drop = Kc - int(round(keep * Kc))
                gd = rnorm2.argsort(1)[:, :drop]
                gm = torch.zeros(T, Kc, device=dev); gm.scatter_(1, gd, 1.0)
                eg = quad(gm, Gram)
                ratio = (eg / (rnorm2 * gm).sum(1).clamp_min(1e-9)).mean().item()  # ||sum||^2/sum||^2
                lm = local_search(gm.clone(), Gram)
                el = quad(lm, Gram)
                res = {"greedy": (eg.sqrt() / ynorm).mean().item(),
                       "local": (el.sqrt() / ynorm).mean().item(),
                       "ortho_ratio": ratio}
                if brute and Kc <= 16:
                    combs = list(itertools.combinations(range(Kc), drop))
                    M = torch.zeros(len(combs), Kc, device=dev)
                    for ii, c in enumerate(combs):
                        M[ii, list(c)] = 1.0
                    cur = torch.full((T,), float('inf'), device=dev)
                    for c0 in range(0, M.shape[0], 4096):
                        Mc = M[c0:c0+4096]
                        e2 = torch.einsum('mi,tij,mj->tm', Mc, Gram, Mc).clamp_min(0)
                        cur = torch.minimum(cur, e2.min(1).values)
                    res["brute"] = (cur.sqrt() / ynorm).mean().item()
                out[keep] = res
            return out

        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
          gc.collect(); torch.cuda.empty_cache(); continue

        rK = run_at(K)
        r16 = run_at(16, brute=True)
        print(f"\n  [{name}] layer {li}, dff={dff}")
        print(f"    -- at K={K} (real granularity) --")
        for keep in KEEPS:
            d = rK[keep]
            print(f"    keep{int(keep*100)}%: greedy {d['greedy']*100:5.1f} | local-search {d['local']*100:5.1f} | "
                  f"ortho-ratio {d['ortho_ratio']:.3f}", flush=True)
        print(f"    -- at G=16 (brute-force validation) --")
        for keep in KEEPS:
            d = r16[keep]
            print(f"    keep{int(keep*100)}%: greedy {d['greedy']*100:5.1f} | local {d['local']*100:5.1f} | "
                  f"BRUTE {d['brute']*100:5.1f} | ortho {d['ortho_ratio']:.3f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: ortho-ratio~1 => residuals orthogonal => greedy provably ~optimal.")
    print("local-search ~ greedy => no headroom even with swaps. local==BRUTE at G16 validates the search.")


if __name__ == "__main__":
    main()
