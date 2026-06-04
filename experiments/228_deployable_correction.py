"""Experiment 228 — DEPLOYABLE low-rank correction of the grouping floor (A). exp227 (oracle ceiling):
a rank-r correction of the dropped-output error closes ~48%(keep50)/83%(keep25) of A. But that used the
TRUE error e(t). Here: PREDICT the correction coords c(t)=B_rᵀe(t) ∈ R^r from the FFN input x(t) (a cheap
always-on rank-r path), add ê=B_r·ĉ to the group-oracle output. Measure how much of the oracle-correction
ceiling SURVIVES prediction. Compare per r: group-oracle (no corr) | +oracle-corr (true e) | +predicted-corr
(from x). Qwen(SwiGLU), keep-pattern grouping K=128, oracle group select, keep50/25.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/228_deployable_correction.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = int(os.environ.get("NCALIB", "8192"))   # reduce for big models (7B) to fit GPU
N_EVAL = 1024
CHUNK = int(os.environ.get("CHUNK", "512"))       # reduce eval-forward memory for big models
K = 128
RMAX = 128
RANKS = [16, 32, 64, 128]
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
            code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
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
            B = cfg['B']
            if cfg.get('neuron'):
                m = keep_topB_neuron(((af.float() - cfg['abar']).abs() * cfg['vn']), B).to(a.dtype)
            else:
                m = oracle_mask(af.float(), cfg['abar'], cfg['vn'], cfg['gsz'], cfg['gf'], B).to(a.dtype)
            masked = af * m + cfg['abar'].to(a.dtype) * (1 - m)
            if cfg.get('corr') == 'oracle':
                STASH[li] = (af - masked).float()
            return (masked.reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            cfg = CFG[li]
            if not cfg['active'] or not cfg.get('corr'):
                return None
            r = cfg['r']; Br = cfg['Br'][:, :r]; sh = output.shape
            if cfg['corr'] == 'oracle':
                e = STASH[li] @ cfg['Wd'].to(STASH[li].device).T; STASH[li] = None
                ehat = (e @ Br) @ Br.T
            else:                                            # predicted from x
                z = (XC[li].float() - cfg['xbar']) @ cfg['P']
                chat = cfg['pred'](z)[:, :r]
                ehat = chat @ Br.T
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

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
    print(f"  dff={dff}\n", flush=True)

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:512].T)
        capa[li] = None; capx[li] = None
        del af; gc.collect(); torch.cuda.empty_cache()

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

    def basis_and_pred(li, bf, gsz, gf):
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; vn = s['vn']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        m = oracle_mask(a, abar, vn, gsz, gf, B)
        drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T                                      # [N,d]
        _, _, Vt = torch.linalg.svd(E, full_matrices=False)
        Br = Vt[:RMAX].T                                     # [d,RMAX]
        c = E @ Br                                           # [N,RMAX] correction coords (targets)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        pred = mlp_fit(z, c, dev)
        del a, m, drop, E, Vt, c, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def setcfg(bf, GRP, corr=None, r=None, BR=None, PRED=None, neuron=False):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=True, B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=neuron, corr=corr, r=r, Wd=s['Wd'] if corr == 'oracle' else None,
                                Br=BR[li] if BR else None, xbar=s['xbar'], P=s['P'], pred=PRED[li] if PRED else None))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = None; CFG[li]['neuron'] = False

    print(f"  SwiGLU ppl (dense {dense_ppl:.3f}), keep-pattern K={K}, oracle group select\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        setcfg(bf, GRP); base = ppl(); off()
        setcfg(bf, GRP, neuron=True); nu = ppl(); off()
        print(f"  keep{int(bf*100)}: group-oracle {base:.3f} | neu-oracle {nu:.3f} | dense {dense_ppl:.3f}", flush=True)
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = basis_and_pred(li, bf, *GRP[li])
        print(f"  {'r':>4} | {'+oracle-corr':>12} | {'+predicted-corr':>15}", flush=True)
        for r in RANKS:
            setcfg(bf, GRP, corr='oracle', r=r, BR=BR); oc = ppl(); off()
            setcfg(bf, GRP, corr='pred', r=r, BR=BR, PRED=PRED); pc = ppl(); off()
            print(f"  {r:>4} | {oc:>12.3f} | {pc:>15.3f}", flush=True)
        print("", flush=True)
    print("READ: predicted-corr between group-oracle (no corr) and oracle-corr => how much of the low-rank", flush=True)
    print("output correction is RECOVERABLE from x (deployable). If predicted≈oracle => A is deployably correctable.", flush=True)


if __name__ == "__main__":
    main()
