"""Experiment 225 — x-router CAPACITY sweep: is gate's advantage over x-routing an INFORMATION limit
(x can't cheaply yield the keep-set) or just under-training of the x-router? exp208 already showed the
DATA axis saturates (~8K tokens); here we sweep the x-router's CAPACITY (PCA features x hidden x depth x
steps). If even a large x-router plateaus below gate => x is information-insufficient at deployable cost
(gate's value is real). If a bigger x-router reaches gate => earlier x-router was just under-capacity.
Qwen(SwiGLU), keep-pattern grouping K=128, keep50/25. References: oracle (true contrib), gate
(|SiLU(g·x)|*ebar*||v||, 1/3 FFN), neu-oracle (per-neuron ceiling).
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/225_xrouter_capacity.py
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
KEEPS = [0.50, 0.25]
# x-router capacity configs: (name, rfeat, hidden, depth, steps)
XR = [("S", 128, 256, 1, 2000), ("M", 512, 512, 1, 4000),
      ("L", 1536, 2048, 1, 8000), ("XL", 1536, 2048, 2, 8000)]


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


def mlp_fit(X, Y, dev, steps, hidden, depth=1, lr=3e-3, bs=2048):
    mods = [nn.Linear(X.shape[1], hidden), nn.GELU()]
    for _ in range(depth - 1):
        mods += [nn.Linear(hidden, hidden), nn.GELU()]
    mods += [nn.Linear(hidden, Y.shape[1])]
    net = nn.Sequential(*mods).to(dev).float()
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
        sc = ((a.float() - cfg['abar']).abs() * vn)
        m = keep_topB_neuron(sc, B).to(a.dtype)
        return a * m + rep.to(a.dtype) * (1 - m)
    Kc = cfg['gsz'].shape[0]
    if mode == 'oracle':
        per = ((a.float() - cfg['abar']) * vn) ** 2
        sg = torch.zeros(n, Kc, device=a.device).index_add_(1, cfg['gf'], per)
    elif mode == 'gate':
        h = F.silu(x.float() @ cfg['Wg'].T)
        per = (h.abs() * cfg['ebar'] * vn) ** 2
        sg = torch.zeros(n, Kc, device=a.device).index_add_(1, cfg['gf'], per)
    else:
        z = (x.float() - cfg['xbar']) @ cfg['P']
        sg = cfg['router'](z).float()
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
    gates = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    print(f"  MODEL={MODEL} [SwiGLU] layers={nL} K={K}", flush=True)
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
        gates[li].register_forward_pre_hook(gpre(li))
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
    hs = [gates[li].register_forward_pre_hook(
        (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li))
        for li in range(nL)]
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    print("  harvested x; building structure + targets", flush=True)

    STR = {}; dff = None
    for li in range(nL):
        x = torch.cat(capx[li]).float().to(dev)
        Wg = layers[li].mlp.gate_proj.weight.detach().float().to(dev)
        Wu = layers[li].mlp.up_proj.weight.detach().float().to(dev)
        Wd = layers[li].mlp.down_proj.weight.detach().float().to(dev)
        dff = Wg.shape[0]; vn = Wd.norm(dim=0)
        h = F.silu(x @ Wg.T); u = x @ Wu.T; a = h * u
        abar = a.mean(0); ebar = u.abs().mean(0)
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        P = Vt.T                                              # [hidden, hidden] full PCA basis
        STR[li] = dict(x=x.half().cpu(), Wg=Wg.cpu(), abar=abar, ebar=ebar, vn=vn,
                       xbar=xbar, P=P, Wd=Wd.cpu(), a=a.half().cpu())
        capx[li] = None
        del x, Wg, Wu, Wd, h, u, a; gc.collect(); torch.cuda.empty_cache()
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

    def rn_target(li, gf):
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev)
        dev_a = a - abar; rn = torch.zeros(a.shape[0], K, device=dev)
        for g in range(K):
            ix = (gf == g).nonzero().flatten()
            if len(ix):
                rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
        del a, dev_a, Wd; gc.collect(); torch.cuda.empty_cache()
        return rn

    def setref(bf, mode, GRP, neuron=False):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=True, B=B, mode=mode, neuron=neuron, rep=s['abar'], abar=s['abar'],
                                vn=s['vn'], gsz=gsz, gf=gf, Wg=s['Wg'].to(dev), ebar=s['ebar'],
                                xbar=s['xbar'], P=s['P'][:, :512]))

    def set_xrouter(bf, GRP, routers, rfeat):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=True, B=B, mode='xrouter', neuron=False, rep=s['abar'],
                                vn=s['vn'], gsz=gsz, gf=gf, xbar=s['xbar'], P=s['P'][:, :rfeat], router=routers[li]))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['neuron'] = False

    print(f"  SwiGLU ppl (dense {dense_ppl:.3f}), keep-pattern K={K}, grp+mean\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        RN = {li: rn_target(li, GRP[li][1]) for li in range(nL)}
        setref(bf, 'oracle', GRP); o = ppl(); off()
        setref(bf, 'gate', GRP); g = ppl(); off()
        setref(bf, 'oracle', GRP, neuron=True); nu = ppl(); off()
        print(f"  keep{int(bf*100)}:  oracle {o:.3f} | gate {g:.3f} | neu-oracle {nu:.3f}", flush=True)
        for name, rfeat, hidden, depth, steps in XR:
            routers = {}
            for li in range(nL):
                s = STR[li]; z = ((s['x'].float().to(dev) - s['xbar']) @ s['P'][:, :rfeat])
                routers[li] = mlp_fit(z, RN[li], dev, steps, hidden, depth)
                del z; gc.collect(); torch.cuda.empty_cache()
            set_xrouter(bf, GRP, routers, rfeat); xr = ppl(); off()
            print(f"            x-router[{name:2} feat{rfeat} h{hidden} d{depth} st{steps}] = {xr:.3f}", flush=True)
        print("", flush=True)
    print("READ: if x-router improves with capacity but PLATEAUS above gate => x is info-insufficient at", flush=True)
    print("cheap cost (gate's value is real, not x-router under-training). If it reaches gate => was under-capacity.", flush=True)


if __name__ == "__main__":
    main()
