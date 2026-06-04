"""Experiment 178 — DEPLOYABLE adaptive-budget (shared-floor): phase-1 deployability gate.
exp176 (ORACLE) found shared-floor + per-token routed-budget concentration beats uniform (keep0.5
sf0.6: 28.55 vs 32.14, +11%). Is that gain LEARNABLE? Train a router to predict per-routed-group
residual; select either UNIFORM (per-token fixed budget) or ADAPTIVE (global threshold tau on
PREDICTED score => variable per-token budget). Compare deploy_uniform vs deploy_thresh (+ oracle
refs) at matched compute. deploy adaptive < deploy uniform => the per-token budget signal is
predictable from x => oracle gain survives deployability => proceed to distill. Qwen-0.5B.
Run: python3 experiments/178_deployable_adaptive_budget.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 2048
N_EVAL = 1024
CHUNK = 512
KROUTE = 64


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
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    a = mlp.act_fn(mlp.gate_proj(xf)) * mlp.up_proj(xf)   # SwiGLU hidden
    N = a.shape[0]; Kc = mlp._gsz.shape[0]
    src, sel = mlp._mode.split("_")                      # src: oracle|deploy ; sel: uniform|global|thresh
    if src == "oracle":
        sq = ((a.float() - mlp._mean.float()).abs() * mlp._vn) ** 2
        score = torch.zeros(N, Kc, device=a.device).index_add_(1, mlp._routed_grp_full, sq)
    else:
        z = (xf.float() - mlp._xbar) @ mlp._P
        score = mlp._router(z).float()
    if sel == "uniform":                                 # per-token fixed routed budget (top route_budget)
        order = score.argsort(1, descending=True); so = mlp._gsz[order]
        keep_ord = (so.cumsum(1) - so) < mlp._route_budget
        selg = torch.zeros(N, Kc, dtype=torch.bool, device=a.device).scatter_(1, order, keep_ord)
    elif sel == "global":                                # exact global top-k across all (token,group) cells
        cost = mlp._gsz.float().unsqueeze(0).expand(N, Kc).reshape(-1)
        o = score.reshape(-1).argsort(descending=True)
        cum = cost[o].cumsum(0) - cost[o]
        kf = torch.zeros(N * Kc, dtype=torch.bool, device=a.device); kf[o] = cum < (N * mlp._route_budget)
        selg = kf.reshape(N, Kc)
    else:                                                # thresh: deployable adaptive — score >= calibrated tau
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
    code = []                                                # CODE corpus (mbpp) — domain-matched to coder models
    for sp in ["train", "test", "validation", "prompt"]:
        try:
            code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    print(f"  MODEL={MODEL}  code tokens={ids.numel()}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)

    @torch.no_grad()
    def ce_eval():
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return tot / ntok

    torch.set_grad_enabled(False)
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        mlp = layers[li].mlp
        hs.append(mlp.gate_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
        hs.append(mlp.down_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"Qwen2.5-0.5B: {nL} layers, dff={dff}", flush=True)
    LST = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        Wup = layers[li].mlp.gate_proj.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        Wd = layers[li].mlp.down_proj.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        abar = a.mean(0); dev_a = a - abar
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        P = Vt[:min(512, Vt.shape[0])].T                       # up to 512 PCA dims
        contrib = dev_a.abs() * Wd.norm(dim=0)
        LST[li] = dict(Wup=Wup.cpu(), Wd=Wd.cpu(), abar=abar, dev_a=dev_a.cpu(), contrib=contrib.cpu(),
                       Z=((x - xbar) @ P).float().cpu(), xbar=xbar.float(), P=P.float())
        layers[li].mlp._mean = abar.to(model.dtype)
        del x, a, dev_a, contrib, Wup, Wd; gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0:
            print(f"  prepped {li}", flush=True)
    tr = slice(0, N_CALIB)

    def configure(bf, sf, strong=False, shmode="neuron"):
        B = int(round(bf * dff))
        rfeat = 512 if strong else 128
        hidden = 512 if strong else 128
        steps = 2500 if strong else 800
        for li in range(nL):
            s = LST[li]
            Wup = s['Wup'].to(dev); Wd = s['Wd'].to(dev); dev_a = s['dev_a'].to(dev); contrib = s['contrib'].to(dev)
            topB = contrib.argsort(1, descending=True)[:, :B]
            freq = torch.zeros(dff, device=dev); freq.scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev))
            n_shared = int(round(sf * B))
            if shmode == "group":          # G-MoE-then-shared: cluster ALL into K, frequent GROUPS -> shared
                grp_all = kmeans(F.normalize(Wup, dim=1), KROUTE, seed=0)
                gfreq = torch.zeros(KROUTE, device=dev).scatter_add_(0, grp_all, freq)
                shared_mask = torch.zeros(dff, device=dev); cum = 0
                for g in gfreq.argsort(descending=True).tolist():
                    if cum >= n_shared:
                        break
                    ig = (grp_all == g); shared_mask[ig] = 1.0; cum += int(ig.sum().item())
                routed_pool = (shared_mask < 0.5).nonzero().flatten()
                rg = grp_all[routed_pool]; uniq = torch.unique(rg)
                remap = torch.full((KROUTE,), -1, dtype=torch.long, device=dev)
                remap[uniq] = torch.arange(len(uniq), device=dev)
                grp_local = remap[rg]
            else:                          # OURS: per-NEURON top keep-frequency -> shared; re-cluster routed
                shared_idx = freq.topk(n_shared).indices if n_shared > 0 else torch.tensor([], dtype=torch.long, device=dev)
                shared_mask = torch.zeros(dff, device=dev); shared_mask[shared_idx] = 1.0
                routed_pool = (shared_mask < 0.5).nonzero().flatten()
                grp_local = kmeans(F.normalize(Wup[routed_pool], dim=1), min(KROUTE, len(routed_pool)), seed=0)
            route_budget = B - int(shared_mask.sum().item())
            Kc = int(grp_local.max().item()) + 1
            gsz = torch.zeros(Kc, device=dev); grp_full = torch.zeros(dff, dtype=torch.long, device=dev)
            routed_is = torch.zeros(dff, device=dev); rn = torch.zeros(dev_a.shape[0], Kc, device=dev)
            for g in range(Kc):
                ix = routed_pool[(grp_local == g).nonzero().flatten()]
                gsz[g] = len(ix); grp_full[ix] = g; routed_is[ix] = 1.0
                if len(ix):
                    rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
            rf = min(rfeat, s['P'].shape[1])
            Ztr = s['Z'][:, :rf].to(dev)
            router = mlp_fit(Ztr[tr], rn[tr], dev, hidden=hidden, steps=steps)
            vn = Wd.norm(dim=0).clone()
            # calibrate tau: global threshold on PREDICTED score to hit avg route_budget neurons/token
            with torch.no_grad():
                pred = router(Ztr[tr]).float()                  # [Ntr, Kc]
            fc = gsz.float().unsqueeze(0).expand_as(pred).reshape(-1)
            fp = pred.reshape(-1); o = fp.argsort(descending=True)
            cum = fc[o].cumsum(0) - fc[o]
            keepn = cum < (pred.shape[0] * route_budget)
            tau = fp[o][keepn].min() if keepn.any() else fp.max()
            del Wup, Wd, dev_a, contrib, rn, Ztr, pred; gc.collect(); torch.cuda.empty_cache()
            mlp = layers[li].mlp
            mlp._xbar = s['xbar']; mlp._P = s['P'][:, :rf].contiguous(); mlp._router = router
            mlp._gsz = gsz.to(model.dtype); mlp._route_budget = float(route_budget); mlp._tau = tau
            mlp._shared_mask = shared_mask.to(model.dtype).unsqueeze(0)
            mlp._routed_grp_full = grp_full; mlp._routed_is = routed_is.to(model.dtype)
            mlp._rep = mlp._mean; mlp._vn = vn; mlp._mode = "deploy_uniform"
        for li in range(nL):
            layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    def set_mode(mode):
        for li in range(nL):
            layers[li].mlp._mode = mode

    def ppl():
        return float(torch.tensor(ce_eval()).exp())

    def meankept():
        return sum(layers[li].mlp._kept for li in range(nL)) / nL

    dense_ce = ce_eval()
    dp = float(torch.tensor(dense_ce).exp())
    print(f"\n  SHARED GRANULARITY: per-NEURON (ours) vs per-GROUP (G-MoE-then-shared). CODE ppl, dense {dp:.2f}")
    print(f"  both shared+global-adaptive sf0.6, training-free.\n")
    print(f"  {'keep':>5s} | {'shared selection':>28s} | {'deploy ppl':>10s} | {'oracle ppl':>10s}")
    print("  " + "-" * 62)
    for bf in [0.85, 0.5]:
        for shmode, tag in [("neuron", "per-NEURON (ours)"), ("group", "per-GROUP (G-MoE-then-shared)")]:
            configure(bf, 0.6, strong=True, shmode=shmode)
            set_mode("oracle_global"); o_or = ppl()
            set_mode("deploy_global"); o_dp = ppl()
            print(f"  {int(bf*100):>4d}% | {tag:>28s} | {o_dp:>10.2f} | {o_or:>10.2f}", flush=True)
        print("  " + "-" * 62)
    print("\nREAD: per-NEURON deploy < per-GROUP => finer per-neuron shared selection beats taking whole")
    print("frequent G-MoE groups as shared. ~equal => group-level is as good (and G-MoE-compatible, simpler).")


if __name__ == "__main__":
    main()
