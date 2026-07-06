"""Experiment 233 — ABLATION of the low-rank output correction's design choices.
Isolates each component of the correction at GROUP-ORACLE selection (so only the correction varies),
Qwen2.5-Coder-1.5B, keep-pattern K128, keep50 & keep25. Reports ppl for:
  floor              : group-oracle, NO correction
  svd+mlp  (default) : SVD-top-r basis, MLP predictor from x, predicted coords
  random+mlp         : RANDOM orthonormal basis (ablate basis: is the low-rank structure learned/real?)
  svd+linear         : SVD basis, LINEAR ridge predictor (ablate predictor: is nonlinearity needed?)
  svd+oracle-coords  : SVD basis, TRUE coords (ablate prediction: upper bound = is error predictable?)
  neuron-oracle      : per-neuron ceiling (reference)
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/233_corr_ablation.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = int(os.environ.get("NCALIB", "8192"))
CHUNK = int(os.environ.get("CHUNK", "512"))
N_EVAL = 1024
K = 128
RCORR = 128
RFEAT = 512
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


def mlp_fit(X, Y, dev, steps=3000, hidden=512, lr=3e-3, bs=2048):
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(0)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def ridge_fit(X, Y, lam=1e-2):
    # closed-form linear ridge: W = (X^T X + lam I)^-1 X^T Y ; returns callable
    XtX = X.T @ X; d = XtX.shape[0]
    W = torch.linalg.solve(XtX + lam * torch.eye(d, device=X.device), X.T @ Y)
    return lambda Z: Z @ W


def keep_topB_neuron(score, B):
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def keep_topB_group(score_g, gsz, gf, B):
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, gf]


def oracle_mask(a, abar, vn, gsz, gf, B):
    per = ((a - abar) * vn) ** 2
    sg = torch.zeros(a.shape[0], gsz.shape[0], device=a.device).index_add_(1, gf, per)
    return keep_topB_group(sg, gsz, gf, B).to(a.dtype)


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
    torch.set_grad_enabled(False)

    CFG = {li: {'active': False} for li in range(nL)}
    XC = {li: None for li in range(nL)}; STASH = {li: None for li in range(nL)}

    def gpre(li):
        def hook(_m, args):
            XC[li] = args[0].reshape(-1, args[0].shape[-1])
        return hook

    def dpre(li):
        def hook(_m, args):
            cfg = CFG[li]
            if not cfg['active']:
                return None
            a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1])
            if cfg.get('neuron'):
                m = keep_topB_neuron(((af.float() - cfg['abar']).abs() * cfg['vn']), cfg['B']).to(a.dtype)
            else:
                m = oracle_mask(af.float(), cfg['abar'], cfg['vn'], cfg['gsz'], cfg['gf'], cfg['B']).to(a.dtype)
            kept = af * m + cfg['abar'].to(a.dtype) * (1 - m)
            if cfg.get('corr') == 'oracle':
                STASH[li] = (af.float() - kept.float())               # dropped activation (for oracle coords)
            return (kept.reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            cfg = CFG[li]
            if not cfg['active'] or not cfg.get('corr'):
                return None
            Br = cfg['Br']; sh = output.shape
            if cfg['corr'] == 'oracle':
                e = STASH[li] @ cfg['Wd'].to(STASH[li].device).T; STASH[li] = None
                ehat = (e @ Br) @ Br.T
            else:                                                     # predicted from x
                z = (XC[li].float() - cfg['xbar']) @ cfg['P']
                ehat = cfg['pred'](z) @ Br.T
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    # ---- harvest ----
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
    dff = capa[0][0].shape[1]

    def ce_eval():
        tot = 0.0; n = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); n += tgt.numel()
        return torch.tensor(tot / n).exp().item()

    # store per-layer calib pieces on CPU
    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del af, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  MODEL={MODEL} dff={dff} N_CALIB={N_CALIB}", flush=True)

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

    def build_basis_pred(li, bf, gsz, gf, basis, predictor):
        """Return (Br[d,r], pred_callable or None). basis in {svd,random}; predictor in {mlp,linear,none}."""
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        m = oracle_mask(a, abar, s['vn'], gsz, gf, B)
        drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T                                               # [N,d]
        d = E.shape[1]
        if basis == 'svd':
            Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
        elif basis == 'rrr':                                          # reduced-rank regression: predictable subspace
            z0 = (s['x'].float().to(dev) - s['xbar']) @ s['P']        # [N,rfeat] features
            coef = torch.linalg.lstsq(z0, E).solution                # [rfeat,d] OLS E~z
            Ehat = (z0 @ coef).cpu()                                  # x-predictable part of E
            _, _, Vt = torch.linalg.svd(Ehat, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
            del z0, coef, Ehat
        else:  # random orthonormal
            g = torch.Generator(device=dev).manual_seed(li)
            Br, _ = torch.linalg.qr(torch.randn(d, RCORR, generator=g, device=dev))
        pred = None
        if predictor != 'none':
            c = E @ Br
            z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
            pred = mlp_fit(z, c, dev) if predictor == 'mlp' else ridge_fit(z, c)
        del a, m, drop, E, Wd; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def setcfg(bf, GRP, mode, Br=None, pred=None):
        # mode: 'off'|'neuron'|'pred'|'oracle'
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=(mode != 'off'), B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=(mode == 'neuron'),
                                corr=('oracle' if mode == 'oracle' else ('pred' if mode == 'pred' else None)),
                                Wd=s['Wd'] if mode == 'oracle' else None,
                                Br=(Br[li] if Br else None), xbar=s['xbar'], P=s['P'],
                                pred=(pred[li] if pred else None)))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = None; CFG[li]['neuron'] = False

    dense = ce_eval(); print(f"  dense ppl {dense:.3f}\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        setcfg(bf, GRP, 'group'); floor = ce_eval(); off()   # group-oracle masking, NO correction (the floor)
        setcfg(bf, GRP, 'neuron'); neu = ce_eval(); off()
        print(f"  === keep{int(bf*100)} (dense {dense:.3f}, floor/group-oracle {floor:.3f}, neuron-oracle {neu:.3f}) ===", flush=True)
        # build the SVD basis + both predictors, and the random basis
        rows = []
        # default: svd + mlp + pred
        Br_svd = {}; pred_mlp = {}
        for li in range(nL):
            Br_svd[li], pred_mlp[li] = build_basis_pred(li, bf, *GRP[li], 'svd', 'mlp')
        setcfg(bf, GRP, 'pred', Br_svd, pred_mlp); rows.append(("svd  + mlp    + x   (default, predicted)", ce_eval())); off()
        # svd + oracle coords (ceiling) -- reuse svd basis, no predictor needed
        setcfg(bf, GRP, 'oracle', Br_svd); rows.append(("svd  + ORACLE coords (ceiling)", ce_eval())); off()
        # svd + linear predictor
        pred_lin = {}
        for li in range(nL):
            _, pred_lin[li] = build_basis_pred(li, bf, *GRP[li], 'svd', 'linear')
        setcfg(bf, GRP, 'pred', Br_svd, pred_lin); rows.append(("svd  + linear + x   (predicted)", ce_eval())); off()
        # random basis + mlp
        Br_rnd = {}; pred_rnd = {}
        for li in range(nL):
            Br_rnd[li], pred_rnd[li] = build_basis_pred(li, bf, *GRP[li], 'random', 'mlp')
        setcfg(bf, GRP, 'pred', Br_rnd, pred_rnd); rows.append(("rand + mlp    + x   (predicted)", ce_eval())); off()
        # RRR basis (predictable subspace) + mlp, and its oracle-coords ceiling
        Br_rrr = {}; pred_rrr = {}
        for li in range(nL):
            Br_rrr[li], pred_rrr[li] = build_basis_pred(li, bf, *GRP[li], 'rrr', 'mlp')
        setcfg(bf, GRP, 'pred', Br_rrr, pred_rrr); rows.append(("RRR  + mlp    + x   (predictable-basis, PRED)", ce_eval())); off()
        setcfg(bf, GRP, 'oracle', Br_rrr); rows.append(("RRR  + ORACLE coords", ce_eval())); off()
        print(f"    floor (no corr)                        {floor:.3f}", flush=True)
        for name, v in rows:
            print(f"    {name:38s} {v:.3f}", flush=True)
        print(f"    neuron-oracle (reference)              {neu:.3f}\n", flush=True)
    print("READ: compares the correction's design choices in isolation (group-oracle selection).", flush=True)
    print("svd>random => the low-rank basis is real/learned; mlp>linear => nonlinearity helps;", flush=True)
    print("predicted vs oracle-coords => how much of the (low-rank) error is recoverable from x.", flush=True)


if __name__ == "__main__":
    main()
