"""Experiment 213 — OPT (ReLU) keep-sweep: adaptive-budget STRUCTURAL win in isolation.
WHY: ReLU has the LARGEST per-token active-count variance (some tokens fire few neurons, some many),
which is exactly what the adaptive per-token budget exploits. BUT ReLU is so sparse that at moderate
keep (50/85) MoEfication is near-lossless for BOTH methods (no gap). So the adaptive win should
appear only at AGGRESSIVE keep (B below the natural active fraction => uniform top-k truncates real
signal). Also ReLU routing gap ~0 (active set predictable) => shared-floor ~0 => any Ours>G-MoE win
on OPT is PURELY adaptive-budget. PREDICTION: oracle-adaptive < oracle-uniform margin GROWS as keep
shrinks (85->50->25->10); deploy~oracle (small routing gap).
OPT has no separate .mlp module (fc1/relu/fc2 live on the decoder layer) => we inject the mask via a
forward_pre_hook on fc2 (its input is the post-ReLU activation a). Metric = PPL (wikitext-2).
Run: HHMODEL=facebook/opt-6.7b python3 experiments/213_opt_keepsweep.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "facebook/opt-6.7b")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
KROUTE = 64
KEEPS = [0.85, 0.50, 0.25, 0.10]


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


def mlp_fit(X, Y, dev, steps=2500, lr=3e-3, bs=2048, hidden=512):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(0)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def select_mask(cfg, a, xf):
    """Return masked activation a*m + rep*(1-m) for the given config dict. a:[N,dff] post-ReLU."""
    N = a.shape[0]; Kc = cfg['gsz'].shape[0]; dev = a.device
    src, sel = cfg['mode'].split("_")
    if src == "oracle":
        sq = ((a.float() - cfg['mean']).abs() * cfg['vn']) ** 2
        score = torch.zeros(N, Kc, device=dev).index_add_(1, cfg['grp_full'], sq)
    else:
        z = (xf.float() - cfg['xbar']) @ cfg['P']
        score = cfg['router'](z).float()
    if sel == "uniform":
        order = score.argsort(1, descending=True); so = cfg['gsz'][order]
        keep_ord = (so.cumsum(1) - so) < cfg['route_budget']
        selg = torch.zeros(N, Kc, dtype=torch.bool, device=dev).scatter_(1, order, keep_ord)
    elif sel == "global":
        cost = cfg['gsz'].float().unsqueeze(0).expand(N, Kc).reshape(-1)
        o = score.reshape(-1).argsort(descending=True)
        cum = cost[o].cumsum(0) - cost[o]
        kf = torch.zeros(N * Kc, dtype=torch.bool, device=dev); kf[o] = cum < (N * cfg['route_budget'])
        selg = kf.reshape(N, Kc)
    else:                                                # thresh: causal per-token adaptive
        selg = score >= cfg['tau']
    m = cfg['shared_mask'].expand(N, -1).clone()
    m = m + selg[:, cfg['grp_full']].to(m.dtype) * cfg['routed_is']
    m = m.clamp(max=1.0)
    cfg['kept'] = m.mean().item()
    return (a * m + cfg['rep'] * (1 - m)).to(a.dtype)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
    print(f"  MODEL={MODEL}  tokens={ids.numel()}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.decoder.layers; nL = len(layers)
    torch.set_grad_enabled(False)

    CFG = {li: {'mode': None, 'active': False} for li in range(nL)}   # per-layer runtime config
    XCACHE = {li: None for li in range(nL)}

    def fc1_pre(li):
        def hook(_m, args):
            XCACHE[li] = args[0].reshape(-1, args[0].shape[-1])
        return hook

    def fc2_pre(li):
        def hook(_m, args):
            cfg = CFG[li]
            if not cfg['active']:
                return None
            a = args[0]; sh = a.shape
            af = a.reshape(-1, sh[-1])
            masked = select_mask(cfg, af, XCACHE[li])
            return (masked.reshape(sh),) + args[1:]
        return hook

    for li in range(nL):
        layers[li].fc1.register_forward_pre_hook(fc1_pre(li))
        layers[li].fc2.register_forward_pre_hook(fc2_pre(li))

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

    dense_ppl = ppl()                                    # hooks inactive => dense
    print(f"  dense ppl {dense_ppl:.3f}", flush=True)

    # ---- harvest structure (hooks return None when inactive, so capture via temp hooks) ----
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(layers[li].fc1.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))(li)))
        hs.append(layers[li].fc2.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"  OPT: {nL} layers, dff={dff}", flush=True)
    # report ReLU sparsity (avg fraction of active neurons per token) to anchor 'aggressive keep'
    spars = float((torch.cat(capa[0]) > 0).float().mean())
    print(f"  layer0 ReLU active fraction ~ {spars:.3f} (keep below this => uniform truncates signal)", flush=True)

    # slim LST: store dev_a in fp16 (CPU); recompute contrib and re-read fc1/fc2 weights in configure
    LST = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        Wd = layers[li].fc2.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        abar = a.mean(0); dev_a = a - abar
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        P = Vt[:min(512, Vt.shape[0])].T
        LST[li] = dict(abar=abar, dev_a=dev_a.half().cpu(),
                       Z=((x - xbar) @ P).float().cpu(), xbar=xbar.float(), P=P.float(), vn=Wd.norm(dim=0).cpu())
        del x, a, dev_a, Wd; gc.collect(); torch.cuda.empty_cache()
    tr = slice(0, N_CALIB)

    def configure(bf, sf):
        B = int(round(bf * dff))
        for li in range(nL):
            s = LST[li]
            Wup = layers[li].fc1.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
            Wd = layers[li].fc2.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
            dev_a = s['dev_a'].float().to(dev); vn = s['vn'].to(dev)
            contrib = dev_a.abs() * vn
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
            Ztr = s['Z'].to(dev)
            router = mlp_fit(Ztr[tr], rn[tr], dev)
            with torch.no_grad():
                pred = router(Ztr[tr]).float()
            fc = gsz.float().unsqueeze(0).expand_as(pred).reshape(-1)
            fp = pred.reshape(-1); o = fp.argsort(descending=True)
            cum = fc[o].cumsum(0) - fc[o]
            keepn = cum < (pred.shape[0] * route_budget)
            tau = fp[o][keepn].min() if keepn.any() else fp.max()
            mean = s['abar'].to(dev)
            CFG[li].update(dict(mean=mean, vn=s['vn'].to(dev), gsz=gsz, route_budget=float(route_budget),
                                tau=tau, shared_mask=shared_mask.unsqueeze(0), grp_full=grp_full,
                                routed_is=routed_is, rep=mean, xbar=s['xbar'].to(dev),
                                P=s['P'].to(dev), router=router, active=True))
            del Wup, Wd, dev_a, contrib, rn, Ztr, pred; gc.collect(); torch.cuda.empty_cache()

    def set_mode(mode):
        for li in range(nL):
            CFG[li]['mode'] = mode

    def set_active(flag):
        for li in range(nL):
            CFG[li]['active'] = flag

    def meankept():
        return sum(CFG[li]['kept'] for li in range(nL)) / nL

    print(f"\n  OPT (ReLU) keep-sweep PPL (dense {dense_ppl:.3f}); SATURATED router 8192\n", flush=True)
    print(f"  {'keep':>5} | {'method':>22} | {'oracle':>9} | {'deploy':>9} | kept", flush=True)
    for bf in KEEPS:
        configure(bf, 0.0)                                   # G-MoE: route-all (sf=0)
        set_active(True)
        set_mode("oracle_uniform"); go = ppl(); kp = meankept()
        set_mode("deploy_uniform"); gd = ppl()
        print(f"  {int(bf*100):>4}% | {'G-MoE (route-all)':>22} | {go:>9.3f} | {gd:>9.3f} | {kp:.3f}", flush=True)
        configure(bf, 0.6)                                   # Ours: shared + adaptive
        set_active(True)
        set_mode("oracle_global"); oo = ppl()
        set_mode("deploy_thresh"); od = ppl()
        print(f"  {int(bf*100):>4}% | {'Ours (shared+adaptive)':>22} | {oo:>9.3f} | {od:>9.3f} |", flush=True)
        set_active(False)
        print("  " + "-" * 60, flush=True)
    print("\nREAD: oracle col is router-independent (adaptive-budget structural ceiling). PREDICTION:", flush=True)
    print("oracle-Ours < oracle-G-MoE margin GROWS as keep shrinks (85->10). deploy~oracle (ReLU routing", flush=True)
    print("easy => shared-floor adds ~0). Any Ours win here is PURELY adaptive budget (no routing-gap help).", flush=True)


if __name__ == "__main__":
    main()
