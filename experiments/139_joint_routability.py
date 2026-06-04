"""Experiment 139 — CONSTRUCT for routability via TRAINING (user pivot, baseline confirmation).
exp 138: cheap re-grouping doesn't beat default. exp 126 (OLMoE) hinted joint training closes
the deployable->oracle gap. Confirm on GeLU baselines + decompose the gap:
  oracle    : true ||r_g|| top-k                         (selection ceiling)
  reg       : router trained on PROXY (MSE to ||r_g||), top-k   (our current deployable)
  e2e       : router trained END-TO-END (straight-through top-k on true FFN recon), FIXED experts
  joint     : e2e router + LEARNABLE representative        (construct expert for routability)
If e2e << reg  => the gap was a bad (proxy) router-training objective, not the experts.
If joint << e2e => expert construction (learnable rep) further helps routability.
All eval by TRUE FFN-output rel-L2 on held-out. mBERT, SantaCoder. K=64, keep 25/50.
Run: python3 experiments/139_joint_routability.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 12288        # exp126 lesson: end-to-end router needs 12k-40k tokens
K = 64
KEEPS = [0.25, 0.5]
SPLIT = 10240
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


def router_net(din, K, dev):
    return torch.nn.Sequential(torch.nn.Linear(din, 256), torch.nn.GELU(),
                               torch.nn.Linear(256, 256), torch.nn.GELU(),
                               torch.nn.Linear(256, K)).to(dev).float()


def err_eval(net, rep, Z, a, Wd, grp, y, ynorm, kk, sl):
    logits = net(Z[sl])
    tk = logits.topk(kk, 1).indices
    km = torch.zeros(logits.shape[0], logits.shape[1], device=Z.device); km.scatter_(1, tk, 1.0)
    mn = km[:, grp]
    a_eff = mn * a[sl] + (1 - mn) * rep
    return ((y[sl] - a_eff @ Wd.T).norm(dim=1) / ynorm[sl]).mean().item() * 100


def train(Z, a, Wd, abar, grp, keep, y, ynorm, learn_rep, steps=3000, lr=5e-4, bs=2048):
    dev = Z.device; K = int(grp.max().item()) + 1; kk = int(round(keep * K))
    net = router_net(Z.shape[1], K, dev)
    rep = abar.clone()
    if learn_rep:
        rep = rep.requires_grad_(True)
    params = list(net.parameters()) + ([rep] if learn_rep else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    gen = torch.Generator(device=dev).manual_seed(0)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, SPLIT, (bs,), generator=gen, device=dev)  # minibatch
            opt.zero_grad()
            logits = net(Z[bi])
            tk = logits.topk(kk, 1).indices
            mh = torch.zeros_like(logits); mh.scatter_(1, tk, 1.0)
            g = torch.sigmoid(logits)
            mask = mh + g - g.detach()                 # straight-through top-k
            mn = mask[:, grp]
            a_eff = mn * a[bi] + (1 - mn) * rep
            loss = ((y[bi] - a_eff @ Wd.T) ** 2).sum(1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)   # exp126: clipping is essential
            opt.step()
    return net.eval(), (rep.detach() if learn_rep else rep)


def mlp_reg(Z, Y, dev, steps=600, lr=4e-3):
    net = router_net(Z.shape[1], Y.shape[1], dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); F.mse_loss(net(Z[:SPLIT]), Y[:SPLIT]).backward(); opt.step()
    return net.eval()


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind in MODELS:
        print(f"\n{'='*66}\n{name}\n{'='*66}", flush=True)
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
          y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6; T = a.shape[0]
          xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
          Z = ((x - xbar) @ Vt[:128].T).float()
          grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
          idxs = [(grp == g).nonzero().flatten() for g in range(K)]
          rn = torch.stack([(dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1) for ix in idxs], 1)
          del model; gc.collect(); torch.cuda.empty_cache()
          ev = slice(SPLIT, T)
          print(f"  layer {li}, dff={dff}, K={K}")
          print(f"  {'keep':>5s} | {'oracle':>7s} | {'reg':>7s} | {'e2e':>7s} | {'joint':>7s}")
          print("  " + "-" * 50)
          for keep in KEEPS:
            kk = int(round(keep * K))
            # oracle
            tk = rn[ev].topk(kk, 1).indices
            km = torch.zeros(rn[ev].shape[0], K, device=dev); km.scatter_(1, tk, 1.0)
            mn = km[:, grp]
            e_or = (((y[ev] - (mn * a[ev] + (1 - mn) * abar) @ Wd.T).norm(dim=1)) / ynorm[ev]).mean().item() * 100
            # reg (proxy)
            reg = mlp_reg(Z, rn, dev)
            e_rg = err_eval(reg, abar, Z, a, Wd, grp, y, ynorm, kk, ev)
            # e2e (fixed experts)
            ne, rpe = train(Z, a, Wd, abar, grp, keep, y, ynorm, learn_rep=False)
            e_e2 = err_eval(ne, rpe, Z, a, Wd, grp, y, ynorm, kk, ev)
            # joint (learnable rep)
            nj, rpj = train(Z, a, Wd, abar, grp, keep, y, ynorm, learn_rep=True)
            e_jt = err_eval(nj, rpj, Z, a, Wd, grp, y, ynorm, kk, ev)
            print(f"  {int(keep*100):>4d}% | {e_or:6.1f}% | {e_rg:6.1f}% | {e_e2:6.1f}% | {e_jt:6.1f}%", flush=True)
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: reg=proxy router(current). e2e=end-to-end straight-through router (fixed experts).")
    print("joint=e2e+learnable representative. If e2e/joint -> oracle, construction-for-routability")
    print("closes the deployable gap (user's lever works). Gap to oracle that remains = intrinsic.")


if __name__ == "__main__":
    main()
