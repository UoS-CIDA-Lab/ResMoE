"""Experiment 140 — Within REMOVAL only: does fine K (low floor) + a DATA-RICH router
give a much lower DEPLOYABLE removal error than G-MoE's K=64?
Known separately: finer K lowers the oracle floor (exp135); more data closes the routing
gap (exp139, K=64 only). Never combined. If deployable @ K=512 << deployable @ K=64, finer
granularity + data is a real deployable removal win. K in {64,256,512}, 12k tokens, keep 25/50.
Caveats (stated, not tested here): fine K erodes MoE hardware efficiency; exp118 found finer-K
fidelity gains did NOT translate to task accuracy. This measures FIDELITY (FFN rel-L2).
Run: python3 experiments/140_fineK_datarich_router.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 12288
SPLIT = 10240
KS = [64, 256, 512]
KEEPS = [0.25, 0.5]
MODELS = [
    ("bert",       "bert-base-multilingual-cased",  "mlm"),
    ("santacoder", "bigcode/gpt_bigcode-santacoder", "causal"),
]


def linears(tag, layer):
    if tag == "bert":
        return layer.intermediate.dense, layer.output.dense
    return layer.mlp.c_fc, layer.mlp.c_proj


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


def mlp_fit(X, Y, dev, steps=1500, lr=3e-3, bs=2048, hidden=256):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(0)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind in MODELS:
        print(f"\n{'='*64}\n{name}\n{'='*64}", flush=True)
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
          a = torch.cat(capa).to(dev)[:N_TOK]; x = torch.cat(capx).to(dev)[:N_TOK]; dff = a.shape[1]
          T = a.shape[0]
          Wd = dn.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
          Wup = up.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
          abar = a.mean(0); dev_a = a - abar
          y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6
          xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
          Z = ((x - xbar) @ Vt[:128].T).float()
          del model; gc.collect(); torch.cuda.empty_cache()
          ev = slice(SPLIT, T)
          # per-unit oracle floor (reference)
          su = dev_a.abs() * Wd.norm(dim=0)
          print(f"  layer {li}, dff={dff}, T={T}")
          for keep in KEEPS:
            ku = int(round(keep * dff)); idx = su[ev].argsort(1, descending=True)
            mk = torch.zeros(su[ev].shape[0], dff, device=dev); mk.scatter_(1, idx[:, :ku], 1.0)
            puf = (((y[ev] - (a[ev] * mk + abar * (1 - mk)) @ Wd.T).norm(dim=1)) / ynorm[ev]).mean().item() * 100
            print(f"    per-unit oracle floor @keep{int(keep*100)}% = {puf:.1f}%")
          print(f"  {'K':>4s} | {'grp':>4s} | {'keep':>5s} | {'oracle':>7s} | {'deployable':>10s} | {'gap':>5s}")
          print("  " + "-" * 56)
          idxs_cache = {}
          for K in KS:
            grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
            idxs = [(grp == g).nonzero().flatten() for g in range(K)]
            rn = torch.zeros(T, K, device=dev)
            for g in range(K):
                ix = idxs[g]
                if ix.numel():
                    rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
            reg = mlp_fit(Z[:SPLIT], rn[:SPLIT], dev)
            sc_ev = reg(Z[ev])

            def dep_err(score, keep):
                kk = int(round(keep * K)); tk = score.argsort(1, descending=True)[:, :kk]
                km = torch.zeros(score.shape[0], K, device=dev); km.scatter_(1, tk, 1.0)
                acc = torch.zeros(score.shape[0], Wd.shape[0], device=dev)
                for g in range(K):
                    ix = idxs[g]
                    if ix.numel():
                        acc += (1 - km[:, g]).unsqueeze(1) * (dev_a[ev][:, ix] @ Wd[:, ix].T)
                return (acc.norm(dim=1) / ynorm[ev]).mean().item() * 100
            for keep in KEEPS:
                eo = dep_err(rn[ev], keep)
                ed = dep_err(sc_ev, keep)
                print(f"  {K:>4d} | {dff//K:>4d} | {int(keep*100):>4d}% | {eo:6.1f}% | {ed:9.1f}% | "
                      f"{ed-eo:4.1f}pt", flush=True)
            del rn; gc.collect(); torch.cuda.empty_cache()
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: does DEPLOYABLE removal error drop a lot K=64->512 (toward the per-unit floor)?")
    print("If yes => finer-K + data is a real deployable removal win (fidelity). gap=routing residual.")


if __name__ == "__main__":
    main()
