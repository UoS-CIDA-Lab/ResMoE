"""Experiment 137 — At finer K (big greedy->optimal gap, exp135), can a DEPLOYABLE router
(predict from x) capture any of it by targeting the OPTIMAL keep-set instead of greedy?
exp 134/136: static can't (per-token). This tests the last route: a learned router from x.
Routers (MLP on 128-PCA of x), all top-k then eval by TRUE per-token error:
  reg        : regress ||r_g||, top-k         (deployable greedy)
  cls-greedy : classify the GREEDY keep-set    (BCE)
  cls-optimal: classify the OPTIMAL keep-set    (BCE)  <- the test
Reference: oracle-greedy (true ||r_g|| top-k) and optimal (local-search) on eval.
If cls-optimal ~ cls-greedy ~ reg  => targeting optimal gives nothing deployable
(the per-token interaction signal is not predictable from x). K in {256,512}.
Run: python3 experiments/137_deployable_optimal_target.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 2048
KS = [256, 512]
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


def ls_chunk(m, Gram, iters=60):
    T, K = m.shape
    diag = torch.diagonal(Gram, dim1=1, dim2=2)
    for _ in range(iters):
        s = torch.einsum('tij,tj->ti', Gram, m)
        delta = (-2 * s + diag).unsqueeze(2) + (2 * s + diag).unsqueeze(1) - 2 * Gram
        inv = (m.unsqueeze(2) < 0.5) | (m.unsqueeze(1) > 0.5)
        best, arg = delta.masked_fill(inv, float('inf')).reshape(T, -1).min(1)
        i = arg // K; j = arg % K
        imp = best < -1e-9
        if not imp.any():
            break
        ti = imp.nonzero().flatten(); m[ti, i[ti]] = 0.0; m[ti, j[ti]] = 1.0
    return m


def local_search(gm, Gram, tc=256):
    out = torch.empty_like(gm)
    for t0 in range(0, gm.shape[0], tc):
        out[t0:t0+tc] = ls_chunk(gm[t0:t0+tc].clone(), Gram[t0:t0+tc])
    return out


def mlp_fit(X, Y, steps=500, lr=4e-3, bce=False, hidden=256):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lf = F.binary_cross_entropy_with_logits if bce else F.mse_loss
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); lf(net(X), Y).backward(); opt.step()
    return net.eval()


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind in MODELS:
        print(f"\n{'='*74}\n{name}\n{'='*74}", flush=True)
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
          print(f"  layer {li}, dff={dff}")
          print(f"  {'K':>4s} | {'keep':>5s} | {'oracle-grd':>10s} | {'optimal':>7s} || "
                f"{'reg':>6s} | {'cls-grd':>7s} | {'cls-opt':>7s}")
          print("  " + "-" * 70)
          for K in KS:
            if K > dff:
                continue
            grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
            idxs = [(grp == g).nonzero().flatten() for g in range(K)]
            Gram = torch.zeros(T, K, K, device=dev)
            for t0 in range(0, T, 128):
                R = torch.stack([dev_a[t0:t0+128, ix] @ Wd[:, ix].T for ix in idxs], 1)
                Gram[t0:t0+128] = torch.einsum('cgd,chd->cgh', R, R)
            rn = torch.diagonal(Gram, dim1=1, dim2=2).clamp_min(0).sqrt()
            for keep in KEEPS:
                drop = K - int(round(keep * K)); kk = K - drop
                gd = (rn * rn).argsort(1)[:, :drop]
                gm = torch.zeros(T, K, device=dev); gm.scatter_(1, gd, 1.0)
                keep_g = 1.0 - gm                          # greedy keep-set
                om = local_search(gm, Gram)
                keep_o = 1.0 - om                          # optimal keep-set
                e_og = (quad(gm[ev], Gram[ev]).sqrt() / ynorm[ev]).mean().item() * 100
                e_op = (quad(om[ev], Gram[ev]).sqrt() / ynorm[ev]).mean().item() * 100
                reg = mlp_fit(Z[tr], rn[tr])
                clg = mlp_fit(Z[tr], keep_g[tr], bce=True)
                clo = mlp_fit(Z[tr], keep_o[tr], bce=True)

                def deploy_err(score):
                    tk = score.argsort(1, descending=True)[:, :kk]
                    km = torch.zeros(score.shape[0], K, device=dev); km.scatter_(1, tk, 1.0)
                    dm = 1 - km
                    return (quad(dm, Gram[ev]).sqrt() / ynorm[ev]).mean().item() * 100
                er = deploy_err(reg(Z[ev])); ecg = deploy_err(clg(Z[ev])); eco = deploy_err(clo(Z[ev]))
                print(f"  {K:>4d} | {int(keep*100):>4d}% | {e_og:9.1f}% | {e_op:6.1f}% || "
                      f"{er:5.1f}% | {ecg:6.1f}% | {eco:6.1f}%", flush=True)
            del Gram; gc.collect(); torch.cuda.empty_cache()
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: deployable routers (reg/cls-grd/cls-opt) vs oracle-greedy & optimal.")
    print("cls-opt ~ cls-grd ~ reg, all >> oracle-greedy => targeting optimal gives nothing")
    print("deployable; the per-token interaction is unpredictable from x. The deployable->oracle")
    print("gap dwarfs the greedy->optimal gap, so chasing optimal is moot.")


if __name__ == "__main__":
    main()
