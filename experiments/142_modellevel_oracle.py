"""Experiment 141 — Resolve the user's paradox: single-layer fidelity improves with finer K
(exp140) but task didn't (exp118). Hypothesis: at MODEL level (all FFNs patched, routing gap
COMPOUNDS through depth, exp115) the finer-K advantage vanishes. Measure MODEL-LEVEL fidelity
(argmax pred-divergence vs unmodified) at K=64/256/512 with DEPLOYABLE routers, Phi-2 (their
GeLU baseline). If model-level pred-div does NOT drop with finer K => single-layer gain doesn't
propagate => no real "fidelity better" at model level => no paradox (and explains exp118 task null).
Run: python3 experiments/141_modellevel_K_phi2.py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "microsoft/phi-2"
N_CALIB = 4096
N_EVAL = 1024
CHUNK = 512
KS = [64, 512]
KEEPS = [0.5, 0.85]
RFEAT = 128


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


def mlp_fit(X, Y, dev, steps=800, lr=3e-3, bs=2048, hidden=128):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(0)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def gm_forward(mlp, x):
    sh = x.shape
    xf = x.reshape(-1, sh[-1])
    a = mlp.activation_fn(mlp.fc1(xf))              # [N, dff] post-gelu
    grp = mlp._gm_grp; gsz = mlp._gm_gsz; Gn = gsz.shape[0]
    if mlp._gm_mode == "oracle":
        sq = ((a.float() - mlp._gm_mean.float()).abs() * mlp._gm_vn) ** 2  # [N,dff]
        score = torch.zeros(a.shape[0], Gn, device=a.device).index_add_(1, grp, sq).to(a.dtype)
    else:
        z = (xf.float() - mlp._gm_xbar) @ mlp._gm_P
        score = mlp._gm_router(z).to(a.dtype)       # [N, Gn]
    budget = mlp._gm_keep * a.shape[1]
    order = score.argsort(1, descending=True)
    so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < budget
    selg = torch.zeros(a.shape[0], Gn, dtype=torch.bool, device=a.device)
    selg.scatter_(1, order, keep_ord)
    m = selg[:, grp].to(a.dtype)
    out = mlp.fc2(a * m + mlp._gm_mean * (1 - m))
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                 trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)
    layers = model.model.layers; nL = len(layers)
    d = model.config.hidden_size

    @torch.no_grad()
    def preds():
        out = []
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            out.append(model(xx).logits[0, :-1].float().argmax(-1).cpu())
        return out
    base = preds()

    # cache per-layer fc1 input x and post-gelu a on calib
    capx = {li: [] for li in range(nL)}
    capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        mlp = layers[li].mlp
        hs.append(mlp.fc1.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
        hs.append(mlp.fc2.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
    with torch.no_grad():
        for c0 in range(0, N_CALIB, CHUNK):
            model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    X = {li: torch.cat(capx[li]) for li in range(nL)}
    A = {li: torch.cat(capa[li]) for li in range(nL)}
    dff = A[0].shape[1]
    print(f"phi-2: {nL} layers, dff={dff}, calib={X[0].shape[0]}", flush=True)

    # precompute per-layer PCA + per-K groups + routers
    store = {}
    for li in range(nL):
        x = X[li].to(dev); a = A[li].to(dev)
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        P = Vt[:RFEAT].T
        z = (x - xbar) @ P
        mean_a = a.mean(0)
        Wd = layers[li].mlp.fc2.weight.detach().float().to(dev)
        Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wup = layers[li].mlp.fc1.weight.detach().float().to(dev)
        Wup = Wup if Wup.shape[0] == dff else Wup.T
        dev_a = a - mean_a
        per_K = {}
        for K in KS:
            grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
            idxs = [(grp == g).nonzero().flatten() for g in range(K)]
            gsz = torch.tensor([len(ix) for ix in idxs], device=dev, dtype=torch.float16)
            rn = torch.zeros(z.shape[0], K, device=dev)
            for g in range(K):
                ix = idxs[g]
                if ix.numel():
                    rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
            router = mlp_fit(z, rn, dev)
            per_K[K] = (grp.to(dev), gsz, router)
        vn = Wd.norm(dim=0).half()
        store[li] = (P.float(), xbar.float(), mean_a.half(), vn, per_K)
        if li % 8 == 0:
            print(f"  built layer {li}", flush=True)
        del x, a, dev_a; gc.collect(); torch.cuda.empty_cache()

    for li in range(nL):
        layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    def set_cfg(K, keep, mode):
        for li in range(nL):
            mlp = layers[li].mlp
            P, xbar, mean_a, vn, per_K = store[li]
            grp, gsz, router = per_K[K]
            mlp._gm_P, mlp._gm_xbar, mlp._gm_mean = P, xbar, mean_a
            mlp._gm_grp, mlp._gm_gsz, mlp._gm_router = grp, gsz, router
            mlp._gm_vn = vn; mlp._gm_keep = keep; mlp._gm_mode = mode

    tot = sum(b.numel() for b in base)
    print(f"\nMODEL-LEVEL argmax pred-divergence vs unmodified phi-2 (eval={tot} toks):\n")
    print(f"  {'mode':>7s} | {'K':>4s} | {'keep':>5s} | pred-div")
    print("  " + "-" * 40)
    for mode in ["oracle", "learned"]:
        for K in KS:
            for keep in KEEPS:
                set_cfg(K, keep, mode)
                ps = preds()
                bad = sum(int((x != y).sum()) for x, y in zip(base, ps))
                print(f"  {mode:>7s} | {K:>4d} | {int(keep*100):>4d}% | {bad/tot*100:6.2f}%", flush=True)
    print("\nREAD: ORACLE finer-K << ORACLE coarse-K (model level) => floor DOES propagate under")
    print("perfect routing => the compounding culprit is the ROUTING gap (bigger at fine K), confirming")
    print("the mechanism. If oracle is also flat => floor itself doesn't propagate (different story).")


if __name__ == "__main__":
    main()
