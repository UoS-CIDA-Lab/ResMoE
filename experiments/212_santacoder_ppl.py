"""Experiment 212 — SantaCoder (GeLU) PPL head-to-head, oracle vs deploy, keep50 + keep85.
WHY: exp159 showed Ours(shared+rt) ppl 60.92 < G-MoE 75.90 on SantaCoder @keep50 — BUT that was
N_CALIB=2048 (under-trained router) + deploy only, so the win mixed (a) adaptive-budget STRUCTURAL
gain and (b) under-training-inflated shared-floor. exp210 (Java, keep85, SATURATED) was noise.
This experiment cleanly separates: SATURATED router (8192), report ORACLE (router-independent,
= adaptive-budget structural ceiling) AND deploy, at BOTH keep50 (aggressive) and keep85 (mild).
KEY TEST: is oracle-Ours < oracle-G-MoE on GeLU @keep50? If yes => the adaptive-budget structural
win is REAL on GeLU too (router-independent), so our scope is broader than "SwiGLU only" — adaptive
helps at aggressive keep regardless of activation; shared-floor adds value only where routing gap.
Metric = PPL (smooth, unlike Java pass@1). Run: python3 experiments/212_santacoder_ppl.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "bigcode/gpt_bigcode-santacoder")
N_CALIB = 8192
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
    a = mlp.act(mlp.c_fc(xf))                             # GeLU FFN hidden (SantaCoder)
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
    elif sel == "thresh":
        selg = score >= mlp._tau
    else:
        order = score.argsort(1, descending=True)
        ssort = score.gather(1, order).clamp_min(0)
        cums = ssort.cumsum(1); tot = cums[:, -1:].clamp_min(1e-9)
        keep_ord = (cums - ssort) < (mlp._topp * tot)
        selg = torch.zeros(N, Kc, dtype=torch.bool, device=a.device).scatter_(1, order, keep_ord)
    m = mlp._shared_mask.expand(N, -1).clone()
    m = m + selg[:, mlp._routed_grp_full].to(m.dtype) * mlp._routed_is
    m = m.clamp(max=1.0)
    mlp._kept = m.mean().item()
    out = mlp.c_proj(a * m + mlp._rep * (1 - m))
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = []
    for sp in ["train", "test", "validation", "prompt"]:
        try:
            code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    print(f"  MODEL={MODEL}  code tokens={ids.numel()}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.transformer.h; nL = len(layers)
    torch.set_grad_enabled(False)

    @torch.no_grad()
    def ce_eval():
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return tot / ntok

    def ppl():
        return float(torch.tensor(ce_eval()).exp())

    dense_ppl = ppl()
    print(f"  dense ppl {dense_ppl:.3f}", flush=True)

    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        mlp = layers[li].mlp
        hs.append(mlp.c_fc.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))(li)))
        hs.append(mlp.c_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"  SantaCoder: {nL} layers, dff={dff}", flush=True)
    LST = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        Wup = layers[li].mlp.c_fc.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        Wd = layers[li].mlp.c_proj.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        abar = a.mean(0); dev_a = a - abar
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        P = Vt[:min(512, Vt.shape[0])].T
        contrib = dev_a.abs() * Wd.norm(dim=0)
        LST[li] = dict(Wup=Wup.cpu(), Wd=Wd.cpu(), abar=abar, dev_a=dev_a.cpu(), contrib=contrib.cpu(),
                       Z=((x - xbar) @ P).float().cpu(), xbar=xbar.float(), P=P.float())
        layers[li].mlp._mean = abar.to(model.dtype)
        del x, a, dev_a, contrib, Wup, Wd; gc.collect(); torch.cuda.empty_cache()
    tr = slice(0, N_CALIB)

    def configure(bf, sf):
        B = int(round(bf * dff)); rfeat = 512; hidden = 512; steps = 2500
        for li in range(nL):
            s = LST[li]
            Wup = s['Wup'].to(dev); Wd = s['Wd'].to(dev); dev_a = s['dev_a'].to(dev); contrib = s['contrib'].to(dev)
            topB = contrib.argsort(1, descending=True)[:, :B]
            freq = torch.zeros(dff, device=dev); freq.scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev))
            n_shared = int(round(sf * B))
            shared_idx = freq.topk(n_shared).indices if n_shared > 0 else torch.tensor([], dtype=torch.long, device=dev)
            shared_mask = torch.zeros(dff, device=dev); shared_mask[shared_idx] = 1.0
            routed_pool = (shared_mask < 0.5).nonzero().flatten()
            route_budget = B - n_shared
            grp_local = kmeans(F.normalize(Wup[routed_pool], dim=1), min(KROUTE, len(routed_pool)), seed=0)
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
            with torch.no_grad():
                pred = router(Ztr[tr]).float()
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

    print(f"\n  SantaCoder (GeLU) PPL head-to-head (dense {dense_ppl:.3f}); SATURATED router 8192\n", flush=True)
    print(f"  {'keep':>5} | {'method':>22} | {'oracle':>8} | {'deploy':>8}", flush=True)
    for bf in [0.85, 0.50]:
        configure(bf, 0.0)                                        # G-MoE: route-all
        set_mode("oracle_uniform"); go = ppl()
        set_mode("deploy_uniform"); gd = ppl()
        print(f"  {int(bf*100):>4}% | {'G-MoE (route-all)':>22} | {go:>8.3f} | {gd:>8.3f}", flush=True)
        configure(bf, 0.6)                                        # Ours: shared + adaptive
        set_mode("oracle_global"); oo = ppl()
        set_mode("deploy_thresh"); od = ppl()
        print(f"  {int(bf*100):>4}% | {'Ours (shared+adaptive)':>22} | {oo:>8.3f} | {od:>8.3f}", flush=True)
        print("  " + "-" * 56, flush=True)
    print("\nREAD: oracle column is router-independent. If oracle-Ours < oracle-G-MoE (esp @keep50) on", flush=True)
    print("this GeLU model => the adaptive-budget STRUCTURAL win is real beyond SwiGLU (scope broader).", flush=True)
    print("deploy gap (oracle->deploy) should be SMALL here (GeLU routing easy) => shared-floor adds little.", flush=True)


if __name__ == "__main__":
    main()
