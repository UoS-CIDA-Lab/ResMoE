"""Experiment 226 — LOW-RANK gate/up approximation for cheap rank-preserving routing. We need only the
RANK (top-B membership), not exact a_k. a_k=SiLU(W_g[k]·x)·(W_u[k]·x). Approximate W_g,W_u by rank-r SVD:
hat_a_k = SiLU((W_g^r x)_k)·(W_u^r x)_k, cost ~2(d+dff)r (vs full gate+up 2·d·dff). DETERMINISTIC (SVD of
existing weights) => NO overfitting (the learned-MLP x-router's failure mode). Rank by hat_a, group-select,
mask the TRUE a. Sweep r; compare to oracle (true a), neu-oracle (ceiling), x-router (learned, ref).
Cost@Qwen: routing ≈ r·5e-4 of FFN (r=64 ≈ 3.3%); skip ALL of gate+up+down on dropped => keep50 save
≈(1-f)-overhead ≈ 50%-3%. If small r ≈ full-gate/oracle => cheap rank-preserving routing dominates.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/226_lowrank_gate.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
K = 128
RMAX = 256
RANKS = [16, 32, 64, 128, 256]
KEEPS = [0.50, 0.25]


def weighted_kmeans_centroids(X, w, k, iters=20, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                wj = w[m]; c[j] = (X[m] * wj.unsqueeze(1)).sum(0) / wj.sum().clamp(min=1e-6)
    return c


def balanced_assign(X, centroids):
    D = torch.cdist(X, centroids); n, k = D.shape
    cap = (n + k - 1) // k
    pref = D.argsort(1); d12 = D.gather(1, pref[:, :2]); regret = d12[:, 1] - d12[:, 0]
    order = regret.argsort(descending=True).tolist(); pref_l = pref.tolist()
    counts = [0] * k; assign = [0] * n
    for i in order:
        for c in pref_l[i]:
            if counts[c] < cap:
                assign[i] = c; counts[c] += 1; break
    return torch.tensor(assign, device=X.device, dtype=torch.long)


def mlp_fit(X, Y, dev, steps=8000, hidden=2048, lr=3e-3, bs=2048):
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(0)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def keep_topB_neuron(score, B):
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def keep_topB_group(score_g, gsz, gf, B):
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, gf]


def compute_masked(cfg, a, x):
    vn = cfg['vn']; rep = cfg['rep']; B = cfg['B']; mode = cfg['mode']; n = a.shape[0]
    if cfg.get('neuron'):
        m = keep_topB_neuron(((a.float() - cfg['abar']).abs() * vn), B).to(a.dtype)
        return a * m + rep.to(a.dtype) * (1 - m)
    Kc = cfg['gsz'].shape[0]
    if mode == 'oracle':
        per = ((a.float() - cfg['abar']) * vn) ** 2
    elif mode == 'lowrank':
        r = cfg['r']; xf = x.float()
        gh = (xf @ cfg['Vtg'][:r].T * cfg['Sg'][:r]) @ cfg['Ug'][:, :r].T
        uh = (xf @ cfg['Vtu'][:r].T * cfg['Su'][:r]) @ cfg['Uu'][:, :r].T
        ah = F.silu(gh) * uh
        per = ((ah - cfg['abar']) * vn) ** 2
    else:  # xrouter
        z = (x.float() - cfg['xbar']) @ cfg['P']
        sg = cfg['router'](z).float()
        m = keep_topB_group(sg, cfg['gsz'], cfg['gf'], B).to(a.dtype)
        return a * m + rep.to(a.dtype) * (1 - m)
    sg = torch.zeros(n, Kc, device=a.device).index_add_(1, cfg['gf'], per)
    m = keep_topB_group(sg, cfg['gsz'], cfg['gf'], B).to(a.dtype)
    return a * m + rep.to(a.dtype) * (1 - m)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = []
    for sp in ["train", "test", "validation", "prompt"]:
        try:
            code += load_dataset("google-research-datasets/mbpp", "full", split=sp)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    print(f"  MODEL={MODEL} [SwiGLU] layers={nL} K={K} ranks={RANKS}", flush=True)
    torch.set_grad_enabled(False)

    CFG = {li: {'active': False} for li in range(nL)}
    XC = {li: None for li in range(nL)}

    def gpre(li):
        def hook(_m, args):
            XC[li] = args[0].reshape(-1, args[0].shape[-1])
        return hook

    def dpre(li):
        def hook(_m, args):
            cfg = CFG[li]
            if not cfg['active']:
                return None
            a = args[0]; sh = a.shape
            return (compute_masked(cfg, a.reshape(-1, sh[-1]), XC[li]).reshape(sh),) + args[1:]
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))

    @torch.no_grad()
    def ce_eval():
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return tot / ntok

    def ppl():
        return float(torch.tensor(ce_eval()).exp())

    dense_ppl = ppl()
    print(f"  dense ppl {dense_ppl:.3f}", flush=True)

    capx = {li: [] for li in range(nL)}
    hs = [gproj[li].register_forward_pre_hook(
        (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li))
        for li in range(nL)]
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    print("  harvested x; SVD of gate/up weights + grouping + x-router", flush=True)

    STR = {}; dff = None
    for li in range(nL):
        x = torch.cat(capx[li]).float().to(dev)
        Wg = layers[li].mlp.gate_proj.weight.detach().float().to(dev)
        Wu = layers[li].mlp.up_proj.weight.detach().float().to(dev)
        Wd = layers[li].mlp.down_proj.weight.detach().float().to(dev)
        dff = Wg.shape[0]; vn = Wd.norm(dim=0)
        a = F.silu(x @ Wg.T) * (x @ Wu.T); abar = a.mean(0)
        Ug, Sg, Vtg = torch.linalg.svd(Wg, full_matrices=False)   # Wg=[dff,d]
        Uu, Su, Vtu = torch.linalg.svd(Wu, full_matrices=False)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(x=x.half().cpu(), abar=abar, vn=vn, a=a.half().cpu(),
                       Ug=Ug[:, :RMAX].cpu(), Sg=Sg[:RMAX].cpu(), Vtg=Vtg[:RMAX].cpu(),
                       Uu=Uu[:, :RMAX].cpu(), Su=Su[:RMAX].cpu(), Vtu=Vtu[:RMAX].cpu(),
                       xbar=xbar, P=VtX[:512].T, Wd=Wd.cpu())
        capx[li] = None
        del x, Wg, Wu, Wd, a, Ug, Sg, Vtg, Uu, Su, Vtu; gc.collect(); torch.cuda.empty_cache()
    print(f"  dff={dff}\n", flush=True)

    def grouping(li, bf):
        s = STR[li]; a = s['a'].float().to(dev); vn = s['vn']; abar = s['abar']; B = int(round(bf * dff))
        Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
        w = ((a - abar).abs() * vn).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        gsz = torch.zeros(K, device=dev)
        for g in range(K):
            gsz[g] = (gl == g).sum()
        del a, Bind; gc.collect(); torch.cuda.empty_cache()
        return gsz, gl

    def base(bf, GRP):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(B=B, vn=s['vn'], gsz=gsz, gf=gf, rep=s['abar'], abar=s['abar'],
                                Ug=s['Ug'].to(dev), Sg=s['Sg'].to(dev), Vtg=s['Vtg'].to(dev),
                                Uu=s['Uu'].to(dev), Su=s['Su'].to(dev), Vtu=s['Vtu'].to(dev),
                                xbar=s['xbar'], P=s['P']))

    def setmode(mode, r=None, neuron=False, router=None):
        for li in range(nL):
            CFG[li]['active'] = True; CFG[li]['mode'] = mode; CFG[li]['neuron'] = neuron; CFG[li]['r'] = r
            if router is not None:
                CFG[li]['router'] = router[li]

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['neuron'] = False

    print(f"  SwiGLU ppl (dense {dense_ppl:.3f}), keep-pattern K={K}, grp+mean\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        base(bf, GRP)
        setmode('oracle'); o = ppl(); off()
        setmode('oracle', neuron=True); nu = ppl(); off()
        print(f"  keep{int(bf*100)}:  oracle {o:.3f} | neu-oracle {nu:.3f}", flush=True)
        for r in RANKS:
            base(bf, GRP); setmode('lowrank', r=r); v = ppl(); off()
            print(f"           lowrank-gate r={r:>3} (≈{r*5.08e-4*100:.1f}% FFN) = {v:.3f}", flush=True)
        # x-router (large, learned) reference
        routers = {}
        for li in range(nL):
            s = STR[li]; a = s['a'].float().to(dev); Wd = s['Wd'].to(dev); dev_a = a - s['abar']
            rn = torch.zeros(a.shape[0], K, device=dev)
            for g in range(K):
                ix = (GRP[li][1] == g).nonzero().flatten()
                if len(ix):
                    rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
            z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
            routers[li] = mlp_fit(z, rn, dev)
            del a, Wd, dev_a, rn, z; gc.collect(); torch.cuda.empty_cache()
        base(bf, GRP); setmode('xrouter', router=routers); xr = ppl(); off()
        print(f"           x-router (learned, L) = {xr:.3f}\n", flush=True)
    print("READ: if small-r lowrank-gate ≈ oracle/full-gate at a few % FFN => cheap rank-preserving routing", flush=True)
    print("(deterministic, no overfit) closes the routing gap far cheaper than full gate or learned x-router.", flush=True)


if __name__ == "__main__":
    main()
