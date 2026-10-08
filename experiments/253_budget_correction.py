"""Experiment 253 — DEPTH-ADAPTIVE correction rank: redistribute the total low-rank-correction budget
across layers instead of using a fixed per-layer rank. Same pipeline as 234 (group-oracle dense / neuron /
group references + predicted low-rank correction on a causal decoder), but the correction rank r_li is
chosen per layer. RANKMODE=fixed reproduces 234 (RCORR per layer); RANKMODE=budget keeps the SAME total
rank nL*RCORR and waterfills it toward layers whose dropped-output error E is higher effective rank
(measure the per-layer profile first with 252). Equal total cost => isolates the depth-allocation effect.
Run: HHMODEL=facebook/opt-1.3b RANKMODE=budget python3 experiments/253_budget_correction.py
     HHMODEL=facebook/opt-1.3b RANKMODE=fixed  python3 experiments/253_budget_correction.py   # = 234
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.reproducibility import load_mbpp_code
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "facebook/opt-1.3b")
N_CALIB = int(os.environ.get("NCALIB", "8192"))
CHUNK = int(os.environ.get("CHUNK", "512"))
N_EVAL = 1024
K = 128
RCORR = 128
RFEAT = 512
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.50,0.25").split(",")]
RANKMODE = os.environ.get("RANKMODE", "budget")           # budget=redistribute nL*RCORR | fixed=RCORR per layer (=234)
RMIN = int(os.environ.get("RMIN", "16"))                  # per-layer rank floor (budget mode)
RMAX = int(os.environ.get("RMAX", "256"))                 # per-layer rank ceiling (budget mode)


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


def find_ffn(layer):
    """Return (in_proj, out_proj) modules for a causal block, gated or not."""
    mlp = getattr(layer, "mlp", layer)
    if hasattr(mlp, "gate_proj"):                 # SwiGLU (Qwen/Llama)
        return mlp.gate_proj, mlp.down_proj
    if hasattr(mlp, "fc1"):                        # OPT-style
        return mlp.fc1, mlp.fc2
    if hasattr(layer, "fc1"):                      # OPT puts fc1/fc2 on the layer
        return layer.fc1, layer.fc2
    if hasattr(mlp, "c_fc"):                        # GPT-2 style
        return mlp.c_fc, mlp.c_proj
    raise RuntimeError("unknown FFN structure")

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = load_mbpp_code()
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    # locate decoder layers
    base = model.model if hasattr(model, "model") else model
    layers = base.decoder.layers if hasattr(base, "decoder") else base.layers
    nL = len(layers)
    inproj = []; outproj = []
    for li in range(nL):
        ip, op = find_ffn(layers[li]); inproj.append(ip); outproj.append(op)
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
            a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1])
            if cfg.get('neuron'):
                m = keep_topB_neuron(((af.float() - cfg['abar']).abs() * cfg['vn']), cfg['B']).to(a.dtype)
            else:
                m = oracle_mask(af.float(), cfg['abar'], cfg['vn'], cfg['gsz'], cfg['gf'], cfg['B']).to(a.dtype)
            return ((af * m + cfg['abar'].to(a.dtype) * (1 - m)).reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            cfg = CFG[li]
            if not cfg['active'] or not cfg.get('corr'):
                return None
            sh = output.shape
            z = (XC[li].float() - cfg['xbar']) @ cfg['P']
            ehat = cfg['pred'](z) @ cfg['Br'].T                # Br is [d, r_li]: any per-layer rank works
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        inproj[li].register_forward_pre_hook(gpre(li))
        outproj[li].register_forward_pre_hook(dpre(li))
        outproj[li].register_forward_hook(dpost(li))

    # ---- harvest ----
    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(outproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(inproj[li].register_forward_pre_hook(
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

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wd = outproj[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del af, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  MODEL={MODEL} dff={dff} nL={nL} act={getattr(model.config,'activation_function',getattr(model.config,'hidden_act','?'))} RANKMODE={RANKMODE}", flush=True)

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

    def error_svd(li, bf, gsz, gf):
        """Build the group-oracle dropped-output error E and cache its top-RMAX right basis, full singular
        spectrum, and projected targets c_full=E@Vt^T (predictor target for any r<=RMAX is c_full[:, :r],
        so the rank can be chosen AFTER the single SVD)."""
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        m = oracle_mask(a, abar, s['vn'], gsz, gf, B)
        E = (a - (a * m + abar * (1 - m))) @ Wd.T
        del a, m, Wd; gc.collect(); torch.cuda.empty_cache()
        Ec = E.cpu(); _, S, Vt = torch.linalg.svd(Ec, full_matrices=False)
        rmax = min(RMAX, Vt.shape[0])
        Vt_top = Vt[:rmax].contiguous()                       # [rmax, d] fp32 cpu
        c_full = (E @ Vt_top.T.to(dev)).cpu()                 # [N, rmax] fp32 projected targets (= E@Br)
        del E, Ec, Vt; gc.collect(); torch.cuda.empty_cache()
        return Vt_top, S, c_full

    def build_pred(li, r, Vt_top, c_full):
        s = STR[li]
        Br = Vt_top[:r].T.to(dev)                             # [d, r]
        c = c_full[:, :r].to(dev)                             # = E @ Br, no recompute
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        pred = mlp_fit(z, c, dev)
        del c, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def alloc_ranks(SPEC):
        """Per-layer rank. fixed: RCORR each (= 234). budget: redistribute the total R=nL*RCORR by
        waterfilling to a common cumulative-energy threshold tau* (more rank to high-effective-rank
        layers), at equal total cost."""
        lis = sorted(SPEC)
        if RANKMODE != "budget":
            return {li: min(RCORR, len(SPEC[li])) for li in lis}, None
        R = RCORR * len(lis)

        def r_at(S, tau):
            s2 = S.double() ** 2; e = s2.cumsum(0) / s2.sum().clamp(min=1e-30)
            return int((e < tau).sum().item()) + 1

        def cl(r): return max(RMIN, min(RMAX, r))
        def total_at(tau): return sum(cl(r_at(SPEC[li], tau)) for li in lis)
        lo, hi = 0.0, 1.0
        for _ in range(50):                                   # bisection: total_at is monotone in tau
            mid = (lo + hi) / 2
            if total_at(mid) < R: lo = mid
            else: hi = mid
        tau = lo
        alloc = {li: cl(r_at(SPEC[li], tau)) for li in lis}
        cur = sum(alloc.values())                             # exact fixup to R by marginal energy
        nextval = lambda li: (SPEC[li][alloc[li]].item() if alloc[li] < len(SPEC[li]) else 0.0)
        prevval = lambda li: (SPEC[li][alloc[li] - 1].item() if 0 < alloc[li] <= len(SPEC[li]) else 0.0)
        while cur < R:
            cand = [li for li in lis if alloc[li] < min(RMAX, len(SPEC[li]))]
            if not cand: break
            li = max(cand, key=nextval); alloc[li] += 1; cur += 1
        while cur > R:
            cand = [li for li in lis if alloc[li] > RMIN]
            if not cand: break
            li = min(cand, key=prevval); alloc[li] -= 1; cur -= 1
        return alloc, tau

    def setcfg(bf, GRP, mode, Br=None, pred=None):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=(mode != 'off'), B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                Br=(Br[li] if Br else None), xbar=s['xbar'], P=s['P'],
                                pred=(pred[li] if pred else None)))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    dense = ce_eval(); print(f"  dense ppl {dense:.3f}\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        setcfg(bf, GRP, 'group'); floor = ce_eval(); off()
        setcfg(bf, GRP, 'neuron'); neu = ce_eval(); off()
        SPEC = {}; CACHE = {}
        for li in range(nL):
            Vt_top, S, c_full = error_svd(li, bf, *GRP[li])
            SPEC[li] = S; CACHE[li] = (Vt_top, c_full)
        alloc, tau = alloc_ranks(SPEC)
        BR = {}; PRED = {}
        for li in range(nL):
            Vt_top, c_full = CACHE[li]
            BR[li], PRED[li] = build_pred(li, alloc[li], Vt_top, c_full)
        CACHE = None; gc.collect(); torch.cuda.empty_cache()
        rs = [alloc[li] for li in range(nL)]
        taustr = f" tau*={tau:.4f}" if tau is not None else ""
        print(f"  [keep{int(bf*100)}] RANKMODE={RANKMODE} rank: min={min(rs)} mean={sum(rs)/nL:.1f} "
              f"max={max(rs)} total={sum(rs)} (budget={RCORR*nL}){taustr}", flush=True)
        setcfg(bf, GRP, 'corr', BR, PRED); corr = ce_eval(); off()
        print(f"  keep{int(bf*100)}: dense {dense:.3f} | neu-oracle {neu:.3f} | group-oracle {floor:.3f} | +corr {corr:.3f}", flush=True)
    print("\nREAD: compare RANKMODE=budget vs fixed at equal total rank (=nL*RCORR); budget's +corr below fixed's", flush=True)
    print("means depth-adaptive rank allocation helps at equal cost. Measure the per-layer profile with 252 first.", flush=True)


if __name__ == "__main__":
    main()
