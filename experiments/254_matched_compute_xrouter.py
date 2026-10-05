"""Experiment 254 — MATCHED-TOTAL-COMPUTE comparison with the DEPLOYABLE selector. exp231 (equal-compute)
and exp236 (baseline table) compare at ORACLE selection, and the baseline table matches the keep fraction
but not the total cost (the correction adds corr_frac ≈ 3.2% FFN on top). Here the same comparison is run
with the x-router (exp230) next to the oracle, at keep f and at the matched cost f + corr_frac:
  G-MoE grouping (weight k-means + mean)            [f]            baseline
  G-MoE grouping, more neurons                      [f+corr_frac]  baseline at ResMoE's total cost
  keep-pattern grouping                             [f]
  keep-pattern grouping, more neurons               [f+corr_frac]  spend the correction budget on neurons
                                                                   (regrouped + router retrained at f+corr_frac, exp231)
  keep-pattern, more neurons, same grouping+router  [f+corr_frac]  identical grouping and router, larger budget only
  keep-pattern + rank-r correction (= ResMoE)       [f+corr_frac]  spend it on the correction
Besides ppl, the in-situ FFN reconstruction error NMSE = sum||y_hat - y||^2 / sum||y||^2 per layer (over eval tokens),
averaged over layers, y = dense FFN output on the same input, correction included for the ResMoE row) is reported.
corr_frac = (d*rfeat + rfeat*h + h*r + r*d) / (3*d*dff) as in exp231, i.e. the correction is charged its own
PCA projection even though the x-router already computes it (conservative for ResMoE). "More neurons" rows
regroup / retrain the router at f+corr_frac and keep whole groups, so they sit at or slightly above budget;
the realized kept fraction is reported. The correction is trained on each selector's own selection error.
ppl is reported on the first 1024 eval tokens (paper protocol) and on all NEVAL tokens.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/254_matched_compute_xrouter.py
Env: KEEPS (0.50,0.25), NEVAL (32768), SEED (router / correction MLP init + batches), OUT=json path.
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
HID = 512
KEEPS = [float(k) for k in os.environ.get("KEEPS", "0.50,0.25").split(",")]
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", "")
SELS = ['oracle', 'xrouter']
ROWS = [('gmoe', "G-MoE grouping (weight k-means + mean)"), ('gmoe_more', "G-MoE grouping, more neurons"),
        ('kp', "keep-pattern grouping"), ('kp_more', "keep-pattern grouping, more neurons"),
        ('kp_more_same', "keep-pattern, more neurons (same grouping+router)"), ('kp_corr', f"keep-pattern + rank-{RCORR} correction (ResMoE)")]


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


def mlp_fit(X, Y, dev, steps=3000, hidden=HID, lr=3e-3, bs=2048):
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


def sel_mask(sel, a, z, s, g, B):
    """bool keep mask [n,dff]: group-oracle (true contribution) or x-router (MLP on PCA(x))."""
    if sel == 'oracle':
        sg = torch.zeros(a.shape[0], K, device=a.device).index_add_(1, g['gf'], ((a - s['abar']) * s['vn']) ** 2)
    else:
        sg = g['selr'](z).float()
    return keep_topB_group(sg, g['gsz'], g['gf'], B)


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
    assert len(ids) >= N_CALIB + N_EVAL and N_EVAL % CHUNK == 0 and N_EVAL >= 1024, (len(ids), N_EVAL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    torch.set_grad_enabled(False)

    STR = {}
    RUN = dict(sel=None, G=None, B=0, corr=None, neuron=False, kept=[0.0, 0], err2=None, y2=None)
    XC = {li: None for li in range(nL)}
    EHAT = {}

    def gpre(li):
        def hook(_m, args):
            XC[li] = args[0].reshape(-1, args[0].shape[-1])
        return hook

    def dpre(li):
        def hook(_m, args):
            if RUN['sel'] is None:
                return None
            s = STR[li]; a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1])
            if RUN['neuron']:
                m = keep_topB_neuron((af.float() - s['abar']).abs() * s['vn'], RUN['B'])
            else:
                m = sel_mask(RUN['sel'], af.float(), (XC[li].float() - s['xbar']) @ s['P'], s, RUN['G'][li], RUN['B'])
            RUN['kept'][0] += float(m.float().mean()); RUN['kept'][1] += 1
            e = ((af.float() - s['abar']) * ~m) @ s['Wd'].T      # dropped-output error on this (in-situ) input
            if RUN['corr'] is not None:
                Br, pred = RUN['corr'][li]; EHAT[li] = pred((XC[li].float() - s['xbar']) @ s['P']) @ Br.T; e = e - EHAT[li]
            RUN['err2'][li] += float((e * e).sum()); RUN['y2'][li] += float(((af.float() @ s['Wd'].T) ** 2).sum())
            return (torch.where(m, af, s['abar'].to(a.dtype)).reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            if RUN['sel'] is None or RUN['corr'] is None:
                return None
            sh = output.shape
            return (output.reshape(-1, sh[-1]) + EHAT[li].to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    @torch.no_grad()
    def ppl(sel=None, G=None, bf=0.0, corr=None, neuron=False):
        """-> (ppl on the first 1024 eval tokens, ppl on all N_EVAL tokens, realized kept fraction, in-situ FFN NMSE)."""
        RUN.update(sel=sel, G=G, B=int(round(bf * dff)), corr=corr, neuron=neuron, kept=[0.0, 0], err2=[0.0] * nL, y2=[0.0] * nL)
        tot = 0.0; ntok = 0; first = None
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
            if c0 + CHUNK == N_CALIB + 1024:
                first = float(torch.tensor(tot / ntok).exp())
        RUN['sel'] = None
        return (first, float(torch.tensor(tot / ntok).exp()), RUN['kept'][0] / max(RUN['kept'][1], 1),
                sum(e / y for e, y in zip(RUN['err2'], RUN['y2'])) / nL if RUN['y2'] and RUN['y2'][0] else float('nan'))

    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(downs[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(gproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]; hidden = capx[0][0].shape[1]
    corr_frac = (hidden * RFEAT + RFEAT * HID + HID * RCORR + RCORR * hidden) / (3.0 * hidden * dff)
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float(); af = a.float().to(dev)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), Wd=Wd, abar=af.mean(0), vn=Wd.norm(dim=0), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del af, x; gc.collect(); torch.cuda.empty_cache()
    dense = ppl()
    print(f"  MODEL={MODEL} dff={dff} hidden={hidden} K={K} SEED={SEED} NEVAL={N_EVAL} | corr_frac(r{RCORR})="
          f"{corr_frac*100:.2f}% FFN | dense ppl {dense[0]:.3f} (1024) {dense[1]:.3f} (all)", flush=True)

    def build(li, kind, bf=None):
        """grouping (kind: 'kp' keep-pattern at keep bf | 'w' G-MoE weight k-means) + its x-router."""
        s = STR[li]; a = s['a'].float().to(dev); dev_a = a - s['abar']
        if kind == 'w':
            Wg = gproj[li].weight.detach().float()
            gf = balanced_assign(Wg, kmeans_centroids(Wg, K, seed=0))
        else:
            con = dev_a.abs() * s['vn']
            Bind = keep_topB_neuron(con, int(round(bf * dff))).float()
            gf = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), con.mean(0), K, seed=0))
            del con, Bind
        rn = torch.zeros(a.shape[0], K, device=dev)
        for g in range(K):
            ix = (gf == g).nonzero().flatten()
            if len(ix):
                rn[:, g] = (dev_a[:, ix] @ s['Wd'][:, ix].T).norm(dim=1)
        selr = mlp_fit((s['x'].float().to(dev) - s['xbar']) @ s['P'], rn, dev)
        del a, dev_a, rn; gc.collect(); torch.cuda.empty_cache()
        return dict(gf=gf, gsz=torch.bincount(gf, minlength=K).float(), selr=selr)

    def train_corr(li, bf, g, sel):
        """rank-r basis + predictor of the dropped-output error under THIS selector's selection."""
        s = STR[li]; a = s['a'].float().to(dev); z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        m = sel_mask(sel, a, z, s, g, int(round(bf * dff)))
        E = ((a - s['abar']) * ~m) @ s['Wd'].T
        _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
        pred = mlp_fit(z, E @ Br, dev)
        del a, z, m, E, Vt; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    GW = {li: build(li, 'w') for li in range(nL)}           # weight grouping / its router do not depend on keep
    RES = dict(model=MODEL, K=K, rcorr=RCORR, seed=SEED, n_calib=N_CALIB, n_eval=N_EVAL, dff=dff,
               corr_frac=corr_frac, dense=dense[:2], keeps={})
    for bf in KEEPS:
        bf2 = bf + corr_frac
        GP = {li: build(li, 'kp', bf) for li in range(nL)}
        GP2 = {li: build(li, 'kp', bf2) for li in range(nL)}
        R = RES['keeps'][f"{bf}"] = {'neu-oracle': ppl('oracle', GP, bf, neuron=True)}
        print(f"\n  ===== keep{int(bf*100)}  (f={bf:.4f}, matched cost f+corr={bf2:.4f}) | dense {dense[1]:.3f} | "
              f"neu-oracle {R['neu-oracle'][1]:.3f} =====", flush=True)
        print(f"  {'configuration':<50} | {'cost':>6} | " + " | ".join(
            f"{s + ' 1024':>13} {s + ' all':>12} {'kept':>6} {'NMSE':>6}" for s in SELS), flush=True)
        for sel in SELS:
            CORR = {li: train_corr(li, bf, GP[li], sel) for li in range(nL)}
            R[sel] = dict(gmoe=ppl(sel, GW, bf), gmoe_more=ppl(sel, GW, bf2), kp=ppl(sel, GP, bf),
                          kp_more=ppl(sel, GP2, bf2), kp_more_same=ppl(sel, GP, bf2), kp_corr=ppl(sel, GP, bf, corr=CORR))
            del CORR; gc.collect(); torch.cuda.empty_cache()
        for key, name in ROWS:
            cost = "f" if key in ('gmoe', 'kp') else "f+corr"
            print(f"  {name:<50} | {cost:>6} | " + " | ".join(
                f"{R[s][key][0]:>13.3f} {R[s][key][1]:>12.3f} {R[s][key][2]*100:>5.1f}% {R[s][key][3]:>6.3f}" for s in SELS), flush=True)
        del GP, GP2; gc.collect(); torch.cuda.empty_cache()
    if OUT:
        pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(OUT).write_text(json.dumps(RES, indent=1))
        print(f"\n  wrote {OUT}", flush=True)
    print("\nREAD: ResMoE row vs the two 'more neurons' rows = same total FFN cost. ResMoE lower => the correction"
          " is a better use of the budget than extra neurons, for that selector.", flush=True)


if __name__ == "__main__":
    main()
