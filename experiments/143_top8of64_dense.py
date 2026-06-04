"""Experiment 143 — Apply OLMoE's native setting (64 experts, TOP-8 = keep 12.5%) to a DENSE
baseline FFN. K=64, keep in {0.125, 0.25, 0.5}, single mid layer, 12k tokens, data-rich router.
Report oracle (true ||r_g|| top-k) and deployable (reg) FFN rel-L2, + per-unit oracle floor.
Point: a dense FFN was trained to use ALL neurons; forcing 8-of-64 (12.5%) is far more
aggressive than what it can tolerate (unlike OLMoE, which was TRAINED with top-8 routing).
Run: python3 experiments/143_top8of64_dense.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 12288
SPLIT = 10240
K = 64
KEEPS = [0.125, 0.25, 0.5]   # 8/64, 16/64, 32/64
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
        print(f"\n{'='*60}\n{name}  (K={K}, top-8 => keep 12.5%)\n{'='*60}", flush=True)
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
          grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
          idxs = [(grp == g).nonzero().flatten() for g in range(K)]
          rn = torch.zeros(T, K, device=dev)
          for g in range(K):
            ix = idxs[g]
            if ix.numel():
                rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
          reg = mlp_fit(Z[:SPLIT], rn[:SPLIT], dev); sc = reg(Z[ev])
          su = dev_a.abs() * Wd.norm(dim=0)

          def grp_err(score, keep):
            kk = int(round(keep * K)); tk = score.argsort(1, descending=True)[:, :kk]
            km = torch.zeros(score.shape[0], K, device=dev); km.scatter_(1, tk, 1.0)
            acc = torch.zeros(score.shape[0], Wd.shape[0], device=dev)
            for g in range(K):
                ix = idxs[g]
                if ix.numel():
                    acc += (1 - km[:, g]).unsqueeze(1) * (dev_a[ev][:, ix] @ Wd[:, ix].T)
            return (acc.norm(dim=1) / ynorm[ev]).mean().item() * 100

          def unit_floor(keep):
            ku = int(round(keep * dff)); idx = su[ev].argsort(1, descending=True)
            mk = torch.zeros(su[ev].shape[0], dff, device=dev); mk.scatter_(1, idx[:, :ku], 1.0)
            return (((y[ev] - (a[ev] * mk + abar * (1 - mk)) @ Wd.T).norm(dim=1)) / ynorm[ev]).mean().item() * 100

          print(f"  layer {li}, dff={dff}")
          print(f"  {'keep':>6s} | {'active':>7s} | {'oracle(K=64)':>12s} | {'deployable':>10s} | {'per-unit floor':>14s}")
          print("  " + "-" * 64)
          for keep in KEEPS:
            print(f"  {keep*100:>5.1f}% | {int(keep*K):>3d}/64 | {grp_err(rn[ev], keep):>11.1f}% | "
                  f"{grp_err(sc, keep):>9.1f}% | {unit_floor(keep):>13.1f}%", flush=True)
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: top-8/64 (keep 12.5%) on a DENSE FFN -- error vs the milder keep 25/50.")
    print("Dense FFN trained to use ALL neurons => 8-of-64 is catastrophic, unlike OLMoE (trained top-8).")


if __name__ == "__main__":
    main()
