"""Experiment 224 — GATE-based routing: does computing the gate (W_g·x, 1/3 of FFN) close the routing
gap? Spectrum: route from x (cheapest, big gap) -> route on computed GATE (1/3 cost, gap?) -> oracle
(compute gate+up, smallest gap). Test on Qwen(SwiGLU), keep-pattern grouping K=128 (best deployable
construction, exp222), keep50/25. Routing signals for GROUP selection (mask the TRUE activation a,
fill dropped with mean):
  oracle   : true contribution |a_k - abar|*||v_k||              (ceiling for this grouping)
  gate     : |SiLU(g_k·x)| * ebar_k * ||v_k||, ebar=calib mean|up|  (gate-only, 1/3 FFN cost)
  x-router : MLP on PCA(x) -> per-group residual norm            (standard cheapest deploy)
Plus neu-oracle (per-neuron true) as the ultimate (non-deployable) ceiling.
If gate << x-router (toward oracle) => computing the gate first meaningfully closes the routing gap
(worth the 1/3 cost). Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/224_gate_routing.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
K = 128
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
    pref = D.argsort(1)
    d12 = D.gather(1, pref[:, :2]); regret = d12[:, 1] - d12[:, 0]
    order = regret.argsort(descending=True).tolist()
    pref_l = pref.tolist()
    counts = [0] * k; assign = [0] * n
    for i in order:
        for c in pref_l[i]:
            if counts[c] < cap:
                assign[i] = c; counts[c] += 1; break
    return torch.tensor(assign, device=X.device, dtype=torch.long)


def grouping_from_assign(gl, dff, dev):
    gsz = torch.zeros(K, device=dev)
    for g in range(K):
        gsz[g] = (gl == g).sum()
    return gsz, gl.clone()


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
    """a:[n,dff] TRUE activation (down_proj input). x:[n,hidden] FFN input (gate input)."""
    vn = cfg['vn']; rep = cfg['rep']; B = cfg['B']; mode = cfg['mode']
    n = a.shape[0]
    if cfg.get('neuron'):                                   # per-neuron oracle (ceiling)
        sc = ((a.float() - cfg['abar']).abs() * vn)
        keep = keep_topB_neuron(sc, B)
        m = keep.to(a.dtype)
        return a * m + rep.to(a.dtype) * (1 - m)
    Kc = cfg['gsz'].shape[0]
    if mode == 'oracle':
        per = ((a.float() - cfg['abar']) * vn) ** 2
        sg = torch.zeros(n, Kc, device=a.device).index_add_(1, cfg['gf'], per)
    elif mode == 'gate':
        h = F.silu(x.float() @ cfg['Wg'].T)                 # [n,dff] gate activation (1/3 FFN compute)
        per = (h.abs() * cfg['ebar'] * vn) ** 2             # |gate| * mean|up| * ||v|| proxy
        sg = torch.zeros(n, Kc, device=a.device).index_add_(1, cfg['gf'], per)
    else:                                                   # x-router
        z = (x.float() - cfg['xbar']) @ cfg['P']
        sg = cfg['router'](z).float()
    keep = keep_topB_group(sg, cfg['gsz'], cfg['gf'], B)
    m = keep.to(a.dtype)
    return a * m + rep.to(a.dtype) * (1 - m)


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
            out = compute_masked(cfg, a.reshape(-1, sh[-1]), XC[li])
            return (out.reshape(sh),) + args[1:]
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

    # harvest FFN input x (gate input) per layer
    capx = {li: [] for li in range(nL)}
    hs = [gates[li].register_forward_pre_hook(
        (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li))
        for li in range(nL)]
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    hidden = capx[0][0].shape[1]
    print(f"  hidden={hidden}", flush=True)

    STR = {}
    dff = None
    for li in range(nL):
        x = torch.cat(capx[li]).float().to(dev)              # [N, hidden]
        Wg = layers[li].mlp.gate_proj.weight.detach().float().to(dev)   # [dff,hidden]
        Wu = layers[li].mlp.up_proj.weight.detach().float().to(dev)
        Wd = layers[li].mlp.down_proj.weight.detach().float().to(dev)
        dff = Wg.shape[0]; vn = Wd.norm(dim=0)
        h = F.silu(x @ Wg.T); u = x @ Wu.T; a = h * u        # [N,dff]
        abar = a.mean(0); ebar = u.abs().mean(0)
        # keep-pattern (direct) grouping at a reference keep (use 0.5 for the fixed grouping)
        Bref = int(round(0.5 * dff))
        Bind = keep_topB_neuron((a - abar).abs() * vn, Bref).float()
        w = ((a - abar).abs() * vn).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        gsz, gf = grouping_from_assign(gl, dff, dev)
        # x-router features + targets (per-group residual norm)
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        P = Vt[:512].T
        rn = torch.zeros(x.shape[0], K, device=dev)
        dev_a = a - abar
        for g in range(K):
            ix = (gf == g).nonzero().flatten()
            if len(ix):
                rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
        router = mlp_fit(((x - xbar) @ P), rn, dev)
        STR[li] = dict(x=x.half().cpu(), Wg=Wg.cpu(), abar=abar, ebar=ebar, vn=vn,
                       gsz=gsz, gf=gf, xbar=xbar, P=P, router=router)
        del x, Wg, Wu, Wd, h, u, a, dev_a, Bind, rn; gc.collect(); torch.cuda.empty_cache()
    print(f"  dff={dff}; grouping=keep-pattern(ref keep50), x-router trained\n", flush=True)

    def setmode(bf, mode, neuron=False):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]
            CFG[li].update(dict(active=True, B=B, mode=mode, neuron=neuron, rep=s['abar'],
                                abar=s['abar'], vn=s['vn'], gsz=s['gsz'], gf=s['gf'],
                                Wg=s['Wg'].to(dev), ebar=s['ebar'], xbar=s['xbar'], P=s['P'], router=s['router']))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['neuron'] = False

    print(f"  SwiGLU ppl (dense {dense_ppl:.3f}), keep-pattern K={K}, grp+mean — routing signal\n", flush=True)
    print(f"  {'keep':>5} | {'oracle':>8} | {'gate':>8} | {'x-router':>9} | {'neu-oracle':>10}", flush=True)
    for bf in KEEPS:
        setmode(bf, 'oracle'); o = ppl(); off()
        setmode(bf, 'gate'); g = ppl(); off()
        setmode(bf, 'xrouter'); xr = ppl(); off()
        setmode(bf, 'oracle', neuron=True); nu = ppl(); off()
        print(f"  {int(bf*100):>4}% | {o:>8.3f} | {g:>8.3f} | {xr:>9.3f} | {nu:>10.3f}", flush=True)
    print("\nREAD: gate between oracle and x-router => computing the gate (1/3 FFN) closes part of the", flush=True)
    print("routing gap. gate≈oracle => gate value is a near-sufficient routing signal (worth the 1/3 cost,", flush=True)
    print("saving up+down on dropped). gate≈x-router => the gate adds little beyond x.", flush=True)


if __name__ == "__main__":
    main()
