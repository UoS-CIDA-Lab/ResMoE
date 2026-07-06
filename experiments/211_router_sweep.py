"""Experiment 211 — ROUTER-DATA SWEEP: is our deploy gap under-training or intrinsic (SwiGLU)?
Motivation: paper trains the selector on ~5M Wikipedia tokens (5000x1024, 30 ep); we used 8192.
Our Qwen-7B SuperGLUE keep85: dense 0.800, best-select 0.760 (-0.040 structural), G-MoE deploy
0.673 (-0.087 router), Ours deploy 0.699. Paper Phi-2 deploy-vs-dense is only -0.009 (GeLU/parallel).
QUESTION: does the -0.087 router gap SHRINK toward best-select as we feed the router more DISTINCT
tokens (=> under-training, baseline needs strengthening) or PLATEAU well below (=> intrinsic SwiGLU
routing gap, paper's small gap is purely GeLU)? Either answer hardens the paper.

Design (memory-safe): Pass A fixes structure (mean/kmeans/shared-split/PCA) on N_STRUCT=8192.
Pass B streams up to max(SWEEP) tokens, accumulating ONLY compact (Z=PCA feats, rn=per-group
residual norm) in fp16 on CPU (wide activations discarded). Then for each N in SWEEP, train the
router on the (Z,rn) PREFIX and eval deploy SuperGLUE. dense + best-select are router-independent
=> computed once. Qwen2.5-Coder-1.5B (SwiGLU), keep85.
Run: python3 experiments/211_router_sweep.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_STRUCT = 8192                                  # tokens to fix structure (saturates fast)
SWEEP = [8192, 32768, 131072, 393216]            # router-training token counts (8K -> ~0.4M, 48x)
CHUNK = 512
KROUTE = 64
BF = 0.85                                         # keep fraction (paper decoder operating point)
RFEAT = 512
HIDDEN = 512
STEPS = 4000


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


def mlp_fit(X, Y, dev, steps=STEPS, lr=3e-3, bs=2048, hidden=HIDDEN):
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
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    a = mlp.act_fn(mlp.gate_proj(xf)) * mlp.up_proj(xf)
    if getattr(mlp, "_harvest", False):                  # stash RAW (unmasked) a + input for router data
        mlp._a_cache = a.detach(); mlp._x_cache = xf.detach()
    N = a.shape[0]; Kc = mlp._gsz.shape[0]
    src, sel = mlp._mode.split("_")
    if src == "oracle":
        sq = ((a.float() - mlp._mean.float()).abs() * mlp._vn) ** 2
        score = torch.zeros(N, Kc, device=a.device).index_add_(1, mlp._routed_grp_full, sq)
    else:
        z = (xf.float() - mlp._xbar) @ mlp._P
        score = mlp._router(z).float()
    if sel == "uniform":
        order = score.argsort(1, descending=True); so = mlp._gsz[order]
        keep_ord = (so.cumsum(1) - so) < mlp._route_budget
        selg = torch.zeros(N, Kc, dtype=torch.bool, device=a.device).scatter_(1, order, keep_ord)
    elif sel == "global":
        cost = mlp._gsz.float().unsqueeze(0).expand(N, Kc).reshape(-1)
        o = score.reshape(-1).argsort(descending=True)
        cum = cost[o].cumsum(0) - cost[o]
        kf = torch.zeros(N * Kc, dtype=torch.bool, device=a.device); kf[o] = cum < (N * mlp._route_budget)
        selg = kf.reshape(N, Kc)
    else:                                            # thresh: causal per-token adaptive
        selg = score >= mlp._tau
    m = mlp._shared_mask.expand(N, -1).clone()
    m = m + selg[:, mlp._routed_grp_full].to(m.dtype) * mlp._routed_is
    m = m.clamp(max=1.0)
    mlp._kept = m.mean().item()
    out = mlp.down_proj(a * m + mlp._rep * (1 - m))
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # wikitext-103 train: enough tokens for the largest sweep point
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    need = max(SWEEP) + N_STRUCT + CHUNK
    buf, ntok = [], 0
    for t in wt["text"]:
        if not t.strip():
            continue
        e = tok(t, return_tensors="pt").input_ids[0]
        buf.append(e); ntok += e.numel()
        if ntok >= need:
            break
    ids = torch.cat(buf)
    print(f"  MODEL={MODEL}  tokens collected={ids.numel()} (need {need})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    torch.set_grad_enabled(False)

    SG = ["boolq", "cb", "copa", "rte", "wic", "wsc"]
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    def sg_eval(tag):
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=4)
        r = lm_eval.simple_evaluate(model=lm, tasks=SG, num_fewshot=0, verbosity="ERROR")
        accs = {t: (r['results'][t].get('acc,none') or r['results'][t].get('acc')) for t in SG}
        avg = sum(accs.values()) / len(accs)
        print(f"  [{tag}] SG-avg={avg:.4f}  (" + " ".join(f"{t}={accs[t]:.3f}" for t in SG) + ")", flush=True)
        del lm, r; gc.collect(); torch.cuda.empty_cache()
        return avg

    dense_sg = sg_eval("DENSE")

    # ---------- Pass A: fix structure on N_STRUCT tokens ----------
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        mlp = layers[li].mlp
        hs.append(mlp.gate_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))(li)))
        hs.append(mlp.down_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))(li)))
    for c0 in range(0, N_STRUCT, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"  {nL} layers, dff={dff}; structure on {N_STRUCT} tokens", flush=True)

    B = int(round(BF * dff))
    STR = {}                                          # per-layer structure
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        Wd = layers[li].mlp.down_proj.weight.detach().float().to(dev)
        Wd = Wd if Wd.shape[1] == dff else Wd.T        # [hidden, dff]
        Wup = layers[li].mlp.gate_proj.weight.detach().float().to(dev)
        Wup = Wup if Wup.shape[0] == dff else Wup.T
        abar = a.mean(0); vn = Wd.norm(dim=0).clone()
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        P = Vt[:RFEAT].T
        contrib = (a - abar).abs() * vn
        # contrib/Wup used only in build_structure -> keep on CPU; abar/vn/xbar/P/Wd stay on GPU
        STR[li] = dict(abar=abar, vn=vn, xbar=xbar, P=P, Wd=Wd, Wup=Wup.cpu(), contrib=contrib.cpu())
        capx[li] = None; capa[li] = None
        del x, a; gc.collect(); torch.cuda.empty_cache()
    del capx, capa; gc.collect()

    def build_structure(sf):
        """Set shared/routed split + grouping for given shared fraction; returns per-layer group idx."""
        ginfo = {}
        for li in range(nL):
            s = STR[li]
            contrib = s['contrib'].to(dev)             # [N_STRUCT, dff]
            topB = contrib.argsort(1, descending=True)[:, :B]
            freq = torch.zeros(dff, device=dev)
            freq.scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev))
            n_shared = int(round(sf * B))
            shared_idx = freq.topk(n_shared).indices if n_shared > 0 else torch.tensor([], dtype=torch.long, device=dev)
            shared_mask = torch.zeros(dff, device=dev); shared_mask[shared_idx] = 1.0
            routed_pool = (shared_mask < 0.5).nonzero().flatten()
            route_budget = B - n_shared
            Wup = s['Wup'].to(dev)
            grp_local = kmeans(F.normalize(Wup[routed_pool], dim=1), min(KROUTE, len(routed_pool)), seed=0)
            del contrib, Wup
            Kc = int(grp_local.max().item()) + 1
            gsz = torch.zeros(Kc, device=dev); grp_full = torch.zeros(dff, dtype=torch.long, device=dev)
            routed_is = torch.zeros(dff, device=dev); gidx = []
            for g in range(Kc):
                ix = routed_pool[(grp_local == g).nonzero().flatten()]
                gsz[g] = len(ix); grp_full[ix] = g; routed_is[ix] = 1.0; gidx.append(ix)
            mlp = layers[li].mlp
            mlp._mean = s['abar'].to(model.dtype); mlp._vn = s['vn']
            mlp._gsz = gsz.to(model.dtype); mlp._route_budget = float(route_budget)
            mlp._shared_mask = shared_mask.to(model.dtype).unsqueeze(0)
            mlp._routed_grp_full = grp_full; mlp._routed_is = routed_is.to(model.dtype)
            mlp._rep = mlp._mean; mlp._xbar = s['xbar']; mlp._P = s['P']
            mlp.forward = types.MethodType(gm_forward, layers[li].mlp)
            ginfo[li] = dict(gidx=gidx, Kc=Kc, route_budget=route_budget)
        return ginfo

    def set_mode(mode):
        for li in range(nL):
            layers[li].mlp._mode = mode

    def stream_router_data(ginfo, n_router):
        """Pass B: stream n_router tokens, accumulate compact (Z, rn) fp16 on CPU per layer.
        Uses gm_forward's stashed RAW a (down_proj input is masked, so we must use the stash)."""
        for li in range(nL):
            layers[li].mlp._harvest = True
        Z = {li: [] for li in range(nL)}; RN = {li: [] for li in range(nL)}
        base = N_STRUCT                                  # tokens AFTER the structure window
        for c0 in range(base, base + n_router, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            model(xx)
            for li in range(nL):
                s = STR[li]; mlp = layers[li].mlp
                xf = mlp._x_cache.float(); af = mlp._a_cache.float()
                z = (xf - s['xbar']) @ s['P']            # [n, rfeat]
                dev_a = af - s['abar']
                Kc = ginfo[li]['Kc']
                rn = torch.zeros(z.shape[0], Kc, device=dev)
                for g in range(Kc):
                    ix = ginfo[li]['gidx'][g]
                    if len(ix):
                        rn[:, g] = (dev_a[:, ix] @ s['Wd'][:, ix].T).norm(dim=1)
                Z[li].append(z.half().cpu()); RN[li].append(rn.half().cpu())
                mlp._a_cache = None; mlp._x_cache = None
        for li in range(nL):
            layers[li].mlp._harvest = False
            Z[li] = torch.cat(Z[li]); RN[li] = torch.cat(RN[li])
        gc.collect(); torch.cuda.empty_cache()
        return Z, RN

    def train_routers(Z, RN, n):
        for li in range(nL):
            Ztr = Z[li][:n].float().to(dev); Ytr = RN[li][:n].float().to(dev)
            router = mlp_fit(Ztr, Ytr, dev)
            # calibrate tau to hit avg route_budget neurons/token
            with torch.no_grad():
                pred = router(Ztr).float()
            mlp = layers[li].mlp; gsz = mlp._gsz.float(); rb = mlp._route_budget
            fc = gsz.unsqueeze(0).expand_as(pred).reshape(-1); fp = pred.reshape(-1)
            o = fp.argsort(descending=True); cum = fc[o].cumsum(0) - fc[o]
            keepn = cum < (pred.shape[0] * rb)
            mlp._tau = fp[o][keepn].min() if keepn.any() else fp.max()
            mlp._router = router
            del Ztr, Ytr, pred; gc.collect(); torch.cuda.empty_cache()

    print(f"\n  ROUTER-DATA SWEEP (Qwen SwiGLU, keep{int(BF*100)}, dense SG-avg {dense_sg:.4f})", flush=True)
    print(f"  best-select (oracle) is router-independent; deploy varies with router tokens.\n", flush=True)

    results = {"dense": dense_sg}
    for sf, name, omode, dmode in [(0.0, "G-MoE", "oracle_uniform", "deploy_uniform"),
                                   (0.6, "Ours", "oracle_global", "deploy_thresh")]:
        ginfo = build_structure(sf)
        set_mode(omode); bs_acc = sg_eval(f"{name} best-select (oracle)")
        results[f"{name}_bestselect"] = bs_acc
        Z, RN = stream_router_data(ginfo, max(SWEEP))
        for n in SWEEP:
            train_routers(Z, RN, n)
            set_mode(dmode); acc = sg_eval(f"{name} deploy  router_tokens={n}")
            results[f"{name}_deploy_{n}"] = acc
        del Z, RN; gc.collect(); torch.cuda.empty_cache()
        print("  " + "-" * 60, flush=True)

    print("\n=== SUMMARY (SG-avg) ===", flush=True)
    print(f"  dense {results['dense']:.4f}", flush=True)
    for name in ["G-MoE", "Ours"]:
        bs = results[f"{name}_bestselect"]
        row = " ".join(f"{n//1024}K={results[f'{name}_deploy_{n}']:.4f}" for n in SWEEP)
        print(f"  {name}: best-select={bs:.4f} | deploy {row}", flush=True)
    print("\nREAD: if deploy RISES toward best-select with more tokens => under-training (strengthen", flush=True)
    print("baseline). if it PLATEAUS below best-select => intrinsic SwiGLU routing gap (paper's small", flush=True)
    print("gap is GeLU-specific). Either way the shared-floor (Ours) recovers part of the gap.", flush=True)


if __name__ == "__main__":
    main()
