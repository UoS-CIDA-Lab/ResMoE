"""Experiment 258 — GROUPING under the DEPLOYABLE selector. The keep-pattern grouping's advantage is shown at
oracle selection (exp236 / baseline table); here the selector is fixed to the x-router (exp230 pipeline) and
only the expert construction changes, with and without the rank-128 correction (trained on the x-router's own
selection error for that grouping):
  weight        MoEfication: balanced k-means on the input-weight rows
  coact         G-MoEfication-style: balanced weighted k-means on the centered, normalized co-activation profile
  keep-pattern  ResMoE: balanced weighted k-means on the per-token co-keep pattern
Each grouping gets its own x-router (same calibration data, same recipe). Reported per keep: ppl without and
with correction (first 1024 eval tokens = paper protocol, and all NEVAL tokens) and the in-situ FFN NMSE (per-layer ratio of sums, averaged over layers).
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/258_grouping_xrouter.py
Env: KEEPS (0.50,0.25), NEVAL (32768), SEED (router / correction MLP init + batches), OUT=json path.
     SELS (xrouter) selectors to evaluate, e.g. SELS=xrouter,oracle adds group-oracle selection (+ a correction trained
     on the oracle selection's error) for every grouping; GROUPS (weight,coact,keep) subset of groupings; CALIB0 (0) start of the 8192 calibration tokens, e.g. CALIB0=40960
     calibrates on an independent slice after the eval region (eval always = tokens [8192, 8192+NEVAL)).
"""
from __future__ import annotations
import sys, pathlib, gc, os, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = int(os.environ.get("NEVAL", "32768"))
CHUNK = 512
K = 128
RCORR = 128
RFEAT = 512
KEEPS = [float(k) for k in os.environ.get("KEEPS", "0.50,0.25").split(",")]
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", "")
CALIB0 = int(os.environ.get("CALIB0", "0"))
SELS = os.environ.get("SELS", "xrouter").split(",")
GROUPINGS = [g for g in [('weight', "weight k-means (MoEfication)"), ('coact', "co-activation (G-MoEfication)"), ('keep', "keep-pattern (ResMoE)")]
             if g[0] in os.environ.get("GROUPS", "weight,coact,keep").split(",")]


def kmeans_centroids(X, k, iters=20, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return c


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


def mlp_fit(X, Y, dev, steps=3000, hidden=512, lr=3e-3, bs=2048):
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(SEED)
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


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = []
    for sp in ["train", "test", "validation", "prompt"]:
        try:
            code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    assert len(ids) >= max(N_CALIB + N_EVAL, CALIB0 + N_CALIB) and N_EVAL % CHUNK == 0 and N_EVAL >= 1024, (len(ids), N_EVAL)
    assert CALIB0 == 0 or CALIB0 >= N_CALIB + N_EVAL, "independent calibration slice must not overlap the eval region"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    torch.set_grad_enabled(False)

    STR = {}
    RUN = dict(G=None, B=0, corr=None, sel='xrouter', err2=None, y2=None)
    XC = {li: None for li in range(nL)}; EHAT = {}

    def gpre(li):
        def hook(_m, args):
            XC[li] = args[0].reshape(-1, args[0].shape[-1])
        return hook

    def dpre(li):
        def hook(_m, args):
            if RUN['G'] is None:
                return None
            s = STR[li]; g = RUN['G'][li]; a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1])
            z = (XC[li].float() - s['xbar']) @ s['P']
            m = select(RUN['sel'], s, g, af.float(), z, RUN['B'])
            e = ((af.float() - s['abar']) * ~m) @ s['Wd'].T
            if RUN['corr'] is not None:
                Br, pred = RUN['corr'][li]; EHAT[li] = pred(z) @ Br.T; e = e - EHAT[li]
            RUN['err2'][li] += float((e * e).sum()); RUN['y2'][li] += float(((af.float() @ s['Wd'].T) ** 2).sum())
            return (torch.where(m, af, s['abar'].to(a.dtype)).reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            if RUN['G'] is None or RUN['corr'] is None:
                return None
            sh = output.shape
            return (output.reshape(-1, sh[-1]) + EHAT[li].to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    def select(sel, s, g, af, z, B):
        """bool keep mask: x-router (predicted group residual norm) or group-oracle (true contribution)."""
        if sel == 'oracle':
            sg = torch.zeros(af.shape[0], K, device=af.device).index_add_(1, g['gf'], ((af - s['abar']) * s['vn']) ** 2)
        else:
            sg = g['selr'](z).float()
        return keep_topB_group(sg, g['gsz'], g['gf'], B)

    @torch.no_grad()
    def ppl(G=None, bf=0.0, corr=None, sel='xrouter'):
        """-> (ppl on the first 1024 eval tokens, ppl on all N_EVAL tokens, in-situ FFN NMSE)."""
        RUN.update(G=G, B=int(round(bf * dff)), corr=corr, sel=sel, err2=[0.0] * nL, y2=[0.0] * nL)
        tot = 0.0; ntok = 0; first = None
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
            if c0 + CHUNK == N_CALIB + 1024:
                first = float(torch.tensor(tot / ntok).exp())
        RUN['G'] = None
        return first, float(torch.tensor(tot / ntok).exp()), sum(e / y for e, y in zip(RUN['err2'], RUN['y2'])) / nL if RUN['y2'] and RUN['y2'][0] else float('nan')

    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(downs[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(gproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for c0 in range(CALIB0, CALIB0 + N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float(); af = a.float().to(dev)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), Wd=Wd, abar=af.mean(0), vn=Wd.norm(dim=0), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del af, x; gc.collect(); torch.cuda.empty_cache()
    dense = ppl()
    print(f"  MODEL={MODEL} dff={dff} K={K} r={RCORR} SEED={SEED} NEVAL={N_EVAL} | dense ppl {dense[0]:.3f} (1024) "
          f"{dense[1]:.3f} (all) | calibration tokens [{CALIB0}, {CALIB0 + N_CALIB}) | selector = x-router for every row", flush=True)

    def build(li, kind, bf):
        """grouping of the given kind (+ its x-router: MLP on PCA(x) -> per-group residual norm)."""
        s = STR[li]; a = s['a'].float().to(dev); dev_a = a - s['abar']; w = (dev_a.abs() * s['vn']).mean(0)
        if kind == 'weight':
            Wg = gproj[li].weight.detach().float(); gf = balanced_assign(Wg, kmeans_centroids(Wg, K, seed=0))
        else:
            if kind == 'coact':
                feat = dev_a.T.contiguous(); feat = feat / feat.norm(dim=1, keepdim=True).clamp(min=1e-6)
            else:
                feat = keep_topB_neuron(dev_a.abs() * s['vn'], int(round(bf * dff))).float().T.contiguous()
            gf = balanced_assign(feat, weighted_kmeans_centroids(feat, w, K, seed=0)); del feat
        rn = torch.zeros(a.shape[0], K, device=dev)
        for g in range(K):
            ix = (gf == g).nonzero().flatten()
            if len(ix):
                rn[:, g] = (dev_a[:, ix] @ s['Wd'][:, ix].T).norm(dim=1)
        selr = mlp_fit((s['x'].float().to(dev) - s['xbar']) @ s['P'], rn, dev)
        del a, dev_a, rn; gc.collect(); torch.cuda.empty_cache()
        return dict(gf=gf, gsz=torch.bincount(gf, minlength=K).float(), selr=selr)

    def train_corr(li, g, bf, sel='xrouter'):
        s = STR[li]; a = s['a'].float().to(dev); z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        m = select(sel, s, g, a, z, int(round(bf * dff)))
        E = ((a - s['abar']) * ~m) @ s['Wd'].T
        _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
        pred = mlp_fit(z, E @ Br, dev)
        del a, z, m, E, Vt; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    RES = dict(model=MODEL, K=K, rcorr=RCORR, seed=SEED, n_eval=N_EVAL, calib0=CALIB0, dff=dff, dense=dense[:2], keeps={})
    for bf in KEEPS:
        R = RES['keeps'][f"{bf}"] = {}
        print(f"\n  ===== keep{int(bf*100)} (x-router selection) =====", flush=True)
        print(f"  {'grouping':<32} {'sel':>8} | {'ppl 1024':>9} {'ppl all':>9} {'NMSE':>6} | {'+corr 1024':>10} {'+corr all':>9} {'NMSE':>6}", flush=True)
        for kind, name in GROUPINGS:
            G = {li: build(li, kind, bf) for li in range(nL)}
            for sel in SELS:
                nc = ppl(G, bf, sel=sel)
                CORR = {li: train_corr(li, G[li], bf, sel) for li in range(nL)}
                wc = ppl(G, bf, corr=CORR, sel=sel)
                if sel == 'xrouter':
                    R[kind] = dict(nocorr=nc, corr=wc)                 # same keys as the earlier runs
                R[f"{kind}|{sel}"] = dict(nocorr=nc, corr=wc)
                print(f"  {name:<32} {sel:>8} | {nc[0]:>9.3f} {nc[1]:>9.3f} {nc[2]:>6.3f} | {wc[0]:>10.3f} {wc[1]:>9.3f} {wc[2]:>6.3f}", flush=True)
                del CORR; gc.collect(); torch.cuda.empty_cache()
            del G; gc.collect(); torch.cuda.empty_cache()
        if OUT:
            pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(OUT).write_text(json.dumps(RES, indent=1))
    if OUT:
        print(f"\n  wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
