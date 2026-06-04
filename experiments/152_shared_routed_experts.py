"""Experiment 152 — SHARED + ROUTED expert construction (DeepSeek-MoE style) vs G-MoE's
route-everything. Idea: hard-wire the ~always-important neurons as an always-on SHARED expert
(computed exactly, not mean-approximated), and only ROUTE the token-dependent rest. At a fixed
active budget, this frees the deployable router from re-deciding 'obvious' neurons -> smaller
deployable->oracle gap -> potentially beats G-MoE.
Single mid layer (mBERT/SantaCoder), budget=keep 50% of neurons total. Sweep shared fraction
of the budget. shared chosen by per-token-oracle KEEP-FREQUENCY (consistently kept neurons).
Compare DEPLOYABLE FFN rel-L2 (reg router on routed pool) vs shared=0 (pure G-MoE) and oracle.
Run: python3 experiments/152_shared_routed_experts.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 12288
SPLIT = 10240
BUDGET = 0.5            # total active fraction
SHARED_FRAC = [0.0, 0.3, 0.6, 0.9]   # fraction of the budget that is always-on shared
KROUTE = 64
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


def mlp_fit(X, Y, dev, steps=1200, lr=3e-3, bs=2048, hidden=256):
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
        print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
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
          chk = 512 if tag == "bert" else 1024
          for c0 in range(0, ids.shape[0], chk):
            model(ids[c0:c0 + chk].unsqueeze(0).to(dev))
          h1.remove(); h2.remove()
          a = torch.cat(capa).to(dev)[:N_TOK]; x = torch.cat(capx).to(dev)[:N_TOK]; dff = a.shape[1]; T = a.shape[0]
          Wup = up.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
          Wd = dn.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
          abar = a.mean(0); dev_a = a - abar
          y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6
          xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
          Z = ((x - xbar) @ Vt[:128].T).float()
          tr, ev = slice(0, SPLIT), slice(SPLIT, T)
          B = int(round(BUDGET * dff))
          # per-neuron contribution magnitude & per-token oracle top-B -> keep frequency
          contrib = dev_a.abs() * Wd.norm(dim=0)            # [T,dff]
          topB = contrib.argsort(1, descending=True)[:, :B]
          freq = torch.zeros(dff, device=dev)
          freq.scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev))
          freq /= T                                          # keep-frequency per neuron

          def err_of_drop(drop_mask_ev):                     # drop_mask_ev: [Nev,dff] bool
            acc = (dev_a[ev] * drop_mask_ev.float()) @ Wd.T
            return (acc.norm(dim=1) / ynorm[ev]).mean().item() * 100

          print(f"  layer {li}, dff={dff}, budget B={B} ({BUDGET:.0%})")
          print(f"  {'shared%':>7s} | {'oracle':>7s} | {'deployable':>10s}")
          print("  " + "-" * 32)
          for sf in SHARED_FRAC:
            n_shared = int(round(sf * B))
            shared_idx = freq.topk(n_shared).indices if n_shared > 0 else torch.tensor([], dtype=torch.long, device=dev)
            shared_set = torch.zeros(dff, dtype=torch.bool, device=dev); shared_set[shared_idx] = True
            routed_pool = (~shared_set).nonzero().flatten()    # neurons to route
            route_budget = B - n_shared                        # neurons to keep among routed
            # group routed pool
            grp_local = kmeans(F.normalize(Wup[routed_pool], dim=1), min(KROUTE, len(routed_pool)), seed=0)
            Kc = int(grp_local.max().item()) + 1
            # per-token group residual norm (oracle) on routed pool
            rn = torch.zeros(T, Kc, device=dev); gsz = torch.zeros(Kc, device=dev)
            gidx = []
            for g in range(Kc):
                ixl = (grp_local == g).nonzero().flatten()
                ix = routed_pool[ixl]; gidx.append(ix)
                gsz[g] = len(ix)
                if len(ix):
                    rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
            reg = mlp_fit(Z[tr], rn[tr], dev); sc_ev = reg(Z[ev])

            def build_drop(score_ev):
                # keep top groups (by score) until route_budget neurons kept among routed
                order = score_ev.argsort(1, descending=True)
                so = gsz[order]
                keep_ord = (so.cumsum(1) - so) < route_budget
                selg = torch.zeros(score_ev.shape[0], Kc, dtype=torch.bool, device=dev)
                selg.scatter_(1, order, keep_ord)
                # drop mask over all dff: shared never dropped; routed dropped if its group not kept
                dm = torch.ones(score_ev.shape[0], dff, dtype=torch.bool, device=dev)
                dm[:, shared_idx] = False
                for g in range(Kc):
                    if len(gidx[g]):
                        dm[:, gidx[g]] = (~selg[:, g]).unsqueeze(1)
                return dm
            e_or = err_of_drop(build_drop(rn[ev]))
            e_dep = err_of_drop(build_drop(sc_ev))
            print(f"  {sf*100:>6.0f}% | {e_or:>6.1f}% | {e_dep:>9.1f}%", flush=True)
          del model; gc.collect(); torch.cuda.empty_cache()
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: shared>0 deployable < shared=0 (pure G-MoE) => hard-wiring always-on neurons as a")
    print("shared expert improves routability/error at fixed budget => a construction that beats G-MoE.")
    print("shared=0 best => routing everything is already optimal; shared split doesn't help.")


if __name__ == "__main__":
    main()
