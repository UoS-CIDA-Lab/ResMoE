"""Experiment 176 — GRACEFUL MoD via a SHARED floor: the constructive MoEfy x MoDfy synthesis.
exp173/174/175: binary whole-FFN skip (skip->0) is catastrophic & training-locked because EVERY
token needs the FFN's always-on ~25% (no identity tokens). FIX grounded in exp174: that always-on
part IS the shared expert. So skip the TOKEN-SPECIFIC routed pool for easy tokens but KEEP the
shared expert (graceful skip -> shared-only, not zero). Test: shared+routed, allocate the routed
budget UNIFORM (every token route_budget) vs GLOBAL (concentrate on hard tokens; easy tokens get
shared-only = graceful MoD), same avg compute, ORACLE. Compare the global-vs-uniform gain WITH a
shared floor (sf0.6) vs WITHOUT (sf0 = exp172, gain was tiny +2-4%). If the shared floor unlocks
a real global>uniform gain => constructive training-free MoEfy x MoDfy. Qwen-0.5B.
Run: python3 experiments/176_graceful_shared_mod.py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_CALIB = 2048
N_EVAL = 2048
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
    # ORACLE routed-group residual norm (shared neurons separate, always-on)
    sq = ((a.float() - mlp._mean.float()).abs() * mlp._vn) ** 2
    score = torch.zeros(N, Kc, device=a.device).index_add_(1, mlp._routed_grp_full, sq)
    if mlp._mode == "uniform":            # each token gets route_budget routed neurons (+shared)
        order = score.argsort(1, descending=True); so = mlp._gsz[order]
        keep_ord = (so.cumsum(1) - so) < mlp._route_budget
        selg = torch.zeros(N, Kc, dtype=torch.bool, device=a.device)
        selg.scatter_(1, order, keep_ord)
    else:                                 # GLOBAL (graceful MoD): spend N*route_budget routed neurons
        cost = mlp._gsz.float().unsqueeze(0).expand(N, Kc).reshape(-1)  # over ALL tokens; easy tokens
        fs = score.reshape(-1); order = fs.argsort(descending=True)     # get few/0 routed = shared-only
        cum = cost[order].cumsum(0) - cost[order]
        sel = cum < (N * mlp._route_budget)
        keep_flat = torch.zeros(N * Kc, dtype=torch.bool, device=a.device)
        keep_flat[order] = sel
        selg = keep_flat.reshape(N, Kc)
    routed_grp_per_tok = selg.sum(1)                      # # routed GROUPS selected per token
    m = mlp._shared_mask.expand(N, -1).clone()            # always-on shared floor (no cliff)
    m = m + selg[:, mlp._routed_grp_full].to(m.dtype) * mlp._routed_is
    m = m.clamp(max=1.0)
    mlp._kept = m.mean().item()
    if getattr(mlp, "_diag", None) is not None:           # [N,2]: routed-group count, total keep frac
        mlp._diag.append(torch.stack([routed_grp_per_tok.float(),
                                      m.sum(1).float() / m.shape[1]], 1).cpu())
    out = mlp.down_proj(a * m + mlp._rep * (1 - m))
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
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
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False); P = Vt[:128].T
        contrib = dev_a.abs() * Wd.norm(dim=0)
        LST[li] = dict(Wup=Wup.cpu(), Wd=Wd.cpu(), abar=abar, dev_a=dev_a.cpu(), contrib=contrib.cpu(),
                       Z=((x - xbar) @ P).float(), xbar=xbar.float(), P=P.float())
        layers[li].mlp._mean = abar.to(model.dtype)
        del x, a, dev_a, contrib, Wup, Wd; gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0:
            print(f"  prepped {li}", flush=True)
    tr = slice(0, N_CALIB)

    def configure(bf, sf):
        B = int(round(bf * dff))
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
            routed_is = torch.zeros(dff, device=dev)
            for g in range(Kc):
                ix = routed_pool[(grp_local == g).nonzero().flatten()]
                gsz[g] = len(ix); grp_full[ix] = g; routed_is[ix] = 1.0
            vn = Wd.norm(dim=0).clone()
            del Wup, Wd, dev_a, contrib; gc.collect(); torch.cuda.empty_cache()
            mlp = layers[li].mlp
            mlp._xbar = s['xbar']; mlp._P = s['P']
            mlp._gsz = gsz.to(model.dtype); mlp._route_budget = float(route_budget)
            mlp._shared_mask = shared_mask.to(model.dtype).unsqueeze(0)
            mlp._routed_grp_full = grp_full; mlp._routed_is = routed_is.to(model.dtype)
            mlp._rep = mlp._mean; mlp._vn = vn; mlp._dff = dff; mlp._mode = "uniform"
        for li in range(nL):
            layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    def set_mode(mode):
        for li in range(nL):
            layers[li].mlp._mode = mode

    def diag_run():
        for li in range(nL):
            layers[li].mlp._diag = []
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
        D = torch.cat([torch.cat(layers[li].mlp._diag) for li in range(nL)])  # [tokens*layers, 2]
        for li in range(nL):
            layers[li].mlp._diag = None
        return D  # col0 = routed-group count, col1 = total keep frac

    dense_ce = ce_eval()
    print(f"\n  EASY-TOKEN DIAGNOSTIC (global/graceful-MoD allocation, ORACLE). dense ppl {torch.tensor(dense_ce).exp():.2f}")
    for keep, sf in [(0.5, 0.6), (0.75, 0.6)]:
        configure(keep, sf); set_mode("global")
        n_shared_frac = layers[0].mlp._shared_mask.float().mean().item()
        D = diag_run()
        grp, kf = D[:, 0], D[:, 1]
        easy = (grp == 0).float().mean().item()      # tokens with ZERO routed groups = shared-only
        print(f"\n  --- keep={keep} shared_frac={sf} (shared expert = {n_shared_frac*100:.0f}% of FFN) ---")
        print(f"  mean total keep {kf.mean()*100:.1f}%  |  EASY tokens (routed=0, shared-only): {easy*100:.1f}%")
        print(f"  easy-token compute = shared floor = {n_shared_frac*100:.0f}% of FFN")
        print(f"  per-token total-keep distribution: " +
              ", ".join(f"p{p}={torch.quantile(kf, p/100).item()*100:.0f}%" for p in [5, 25, 50, 75, 95]))
        print(f"  routed-group count per token:       " +
              ", ".join(f"p{p}={torch.quantile(grp, p/100).item():.0f}" for p in [5, 25, 50, 75, 95]) +
              f"  (max group idx {int(grp.max().item())})")
    print("\nREAD: EASY% = fraction of tokens the graceful-MoD runs at shared-only (the FFN's always-on")
    print("part); the rest get token-specific routed compute concentrated on them. Wide keep-distribution")
    print("=> real per-token compute heterogeneity being exploited.")


if __name__ == "__main__":
    main()
