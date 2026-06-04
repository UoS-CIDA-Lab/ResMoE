"""Experiment 131 — Port our two core diagnostics to the G-MoEfication paper's OWN
baseline models (all GeLU dense): mBERT, SantaCoder-1.1B, Falcon-7B.
(Phi-2-2.7B, their 4th model, already done in exp 117-122.)

For each model, 3 layers (~1/4, 1/2, 3/4 depth), 2048 tokens. Trick: hook the DOWN
linear's pre-forward to capture the post-activation hidden `a` directly (act-fn agnostic),
and the UP linear's pre-forward to capture the FFN input `x`. down.weight columns = v_k.

(A) MERGE / COMPRESSION WALL:
    - |cosine| nearest-neighbor of up-weight rows (mergeable input directions?)
    - per-token ORACLE drop floor (keep 25/50/85): replace dropped neuron by its mean.
(B) ROOM-TO-IMPROVE (G=16 groups, brute-force tractable):
    - greedy(drop smallest ||r_g||) vs BRUTE-OPTIMAL drop-set; separability (greedy==opt%)
    - deployable routers: regression-to-||r_g|| vs classification-to-optimal-keepset, top-k
Run: python3 experiments/131_gmoe_baseline_models.py
"""
from __future__ import annotations
import sys, pathlib, itertools, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 2048
G = 16
KEEPS = [0.25, 0.5, 0.85]
BF_KEEPS = [0.25, 0.5]      # brute-force + router levels
SPLIT = 1536

MODELS = [
    ("bert",       "bert-base-multilingual-cased", "mlm"),
    ("santacoder", "bigcode/gpt_bigcode-santacoder", "causal"),
    ("falcon",     "tiiuae/falcon-7b",             "causal"),
]


def linears(tag, layer):
    if tag == "bert":
        return layer.intermediate.dense, layer.output.dense
    if tag == "santacoder":
        return layer.mlp.c_fc, layer.mlp.c_proj
    if tag == "falcon":
        return layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h
    raise ValueError(tag)


def layer_list(tag, model):
    if tag == "bert":
        return model.encoder.layer
    return model.transformer.h


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


def mlp_fit(X, Y, steps=400, lr=5e-3, bce=False, hidden=128):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = F.binary_cross_entropy_with_logits if bce else F.mse_loss
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); lossf(net(X), Y).backward(); opt.step()
    return net.eval()


def analyze_ffn(a, x, Wup, Wd, dev, masks):
    """a:[T,dff] post-act hidden, x:[T,d] ffn input, Wup:[dff,d], Wd:[d,dff]."""
    import numpy as np
    T, dff = a.shape
    out = {}
    # (A) merge: |cosine| NN of up-weight rows (input directions)
    n = F.normalize(Wup, dim=1)
    S = n @ n.T; S.fill_diagonal_(0)
    out["nn_abscos"] = S.abs().max(1).values.mean().item()
    out["nn_signed"] = S.max(1).values.mean().item()
    # (A) oracle drop floor
    abar = a.mean(0); dev_a = a - abar
    vnorm = Wd.norm(dim=0)
    score = dev_a.abs() * vnorm
    y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6
    for keep in KEEPS:
        k = int(round(keep * dff))
        idx = score.argsort(1, descending=True)
        mask = torch.zeros_like(a); mask.scatter_(1, idx[:, :k], 1.0)
        a_app = a * mask + abar * (1 - mask)
        out[f"oracle{keep}"] = ((y - a_app @ Wd.T).norm(dim=1) / ynorm).mean().item()
    # (B) G groups via up-weight kmeans
    grp = kmeans(F.normalize(Wup, dim=1), G, seed=0)
    R = torch.stack([dev_a[:, grp == g] @ Wd[:, grp == g].T for g in range(G)], 1)  # [T,G,d]
    Gram = torch.einsum('tgd,thd->tgh', R, R)
    rnorm = torch.diagonal(Gram, dim1=1, dim2=2).clamp_min(0).sqrt()
    xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
    Z = ((x - xbar) @ Vt[:128].T).float()
    tr, ev = slice(0, SPLIT), slice(SPLIT, T)
    for keep in BF_KEEPS:
        drop = G - int(round(keep * G)); M = masks[keep]
        cur = torch.full((T,), float('inf'), device=dev); best = torch.zeros(T, dtype=torch.long, device=dev)
        for c0 in range(0, M.shape[0], 4096):
            Mc = M[c0:c0 + 4096]
            e2 = torch.einsum('mi,tij,mj->tm', Mc, Gram, Mc).clamp_min(0)
            v, j = e2.min(1); upd = v < cur
            cur = torch.where(upd, v, cur); best = torch.where(upd, j + c0, best)
        opt_drop = M[best]; opt_err = cur.sqrt() / ynorm
        gd = rnorm.argsort(1)[:, :drop]
        gm = torch.zeros(T, G, device=dev); gm.scatter_(1, gd, 1.0)
        grd = (torch.einsum('ti,tij,tj->t', gm, Gram, gm).clamp_min(0).sqrt() / ynorm)
        sep = (gm == opt_drop).all(1).float().mean().item()
        keep_opt = 1.0 - opt_drop
        reg = mlp_fit(Z[tr], rnorm[tr]); scl = mlp_fit(Z[tr], keep_opt[tr], bce=True)

        def kerr(sc):
            kk = G - drop; tk = sc.argsort(1, descending=True)[:, :kk]
            km = torch.zeros(sc.shape[0], G, device=dev); km.scatter_(1, tk, 1.0); dm = 1 - km
            Gv = Gram[ev]
            return (torch.einsum('ti,tij,tj->t', dm, Gv, dm).clamp_min(0).sqrt() / ynorm[ev]).mean().item()
        out[f"greedy{keep}"] = grd[ev].mean().item()
        out[f"optimal{keep}"] = opt_err[ev].mean().item()
        out[f"oracleR{keep}"] = kerr(rnorm[ev])
        out[f"reg{keep}"] = kerr(reg(Z[ev]))
        out[f"cls{keep}"] = kerr(scl(Z[ev]))
        out[f"sep{keep}"] = sep
    return out


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    masks = {}
    for keep in BF_KEEPS:
        drop = G - int(round(keep * G))
        combs = list(itertools.combinations(range(G), drop))
        M = torch.zeros(len(combs), G, device=dev)
        for i, c in enumerate(combs):
            M[i, list(c)] = 1.0
        masks[keep] = M
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())

    for tag, name, kind in MODELS:
        print(f"\n{'='*70}\n{name} ({kind})\n{'='*70}", flush=True)
        try:
          tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
          ids = tok(text, return_tensors="pt").input_ids[0][:N_TOK]
          Loader = AutoModel if kind == "mlm" else AutoModelForCausalLM
          model = Loader.from_pretrained(name, dtype=torch.float16,
                                       trust_remote_code=True).to(dev).eval()
          if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
          torch.set_grad_enabled(False)
          layers = layer_list(tag, model); nL = len(layers)
          targets = sorted({nL // 4, nL // 2, 3 * nL // 4})
          capx, capa = {li: [] for li in targets}, {li: [] for li in targets}
          hs = []
          for li in targets:
            up, dn = linears(tag, layers[li])
            hs.append(up.register_forward_pre_hook(
                (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
            hs.append(dn.register_forward_pre_hook(
                (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
          chunk = 512 if tag == "bert" else 1024
          for c0 in range(0, ids.shape[0], chunk):
            model(ids[c0:c0 + chunk].unsqueeze(0).to(dev))
          for h in hs:
            h.remove()
          print(f"  layers {nL}, sampled {targets}", flush=True)
          rows = []
          for li in targets:
            up, dn = linears(tag, layers[li])
            a = torch.cat(capa[li]); x = torch.cat(capx[li])
            dff = a.shape[1]
            Wup = up.weight.detach().float()
            if Wup.shape[0] != dff:
                Wup = Wup.T
            Wd = dn.weight.detach().float()
            if Wd.shape[1] != dff:
                Wd = Wd.T
            r = analyze_ffn(a.to(dev), x.to(dev), Wup.to(dev), Wd.to(dev), dev, masks)
            r["layer"] = li; r["dff"] = dff
            rows.append(r); print(f"    layer {li} (dff={dff}) done", flush=True)
          def mean(k): return float(np.mean([r[k] for r in rows]))
          print(f"\n  [{name}] mean over {len(rows)} layers (dff={rows[0]['dff']}):")
          print(f"    merge: |cos|NN={mean('nn_abscos'):.3f} (signed {mean('nn_signed'):.3f})  "
              f"oracle-drop keep25/50/85 = {mean('oracle0.25')*100:.1f}/{mean('oracle0.5')*100:.1f}/{mean('oracle0.85')*100:.1f}%")
          for keep in BF_KEEPS:
            print(f"    keep{int(keep*100)}%: greedy {mean(f'greedy{keep}')*100:5.1f} | "
                  f"optimal {mean(f'optimal{keep}')*100:5.1f} | oracle {mean(f'oracleR{keep}')*100:5.1f} | "
                  f"reg {mean(f'reg{keep}')*100:5.1f} | cls {mean(f'cls{keep}')*100:5.1f} | "
                  f"greedy==opt {mean(f'sep{keep}')*100:4.1f}%", flush=True)
          del model
        except Exception as ex:
          import traceback; traceback.print_exc()
          print(f"  [SKIP {name}] {type(ex).__name__}: {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nDONE. Compare to OLMoE/SwiGLU: separability (greedy==opt) high => MLP+topk right;")
    print("optimal~greedy => no set-headroom; reg/cls vs greedy => deployable supervision value.")


if __name__ == "__main__":
    main()
