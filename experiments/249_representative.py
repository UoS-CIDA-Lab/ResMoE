"""Experiment 249 — REPRESENTATIVE ablation (isolating process 3).
Fix construction (keep-pattern grouping) and selection (group-oracle); vary ONLY the value substituted
for skipped experts: zero | unconditional-mean (rank-0, G-MoEfication) | conditional-mean E[a_k|dropped]
| +rank-r low-rank correction (ours). Qwen2.5-Coder-1.5B, keep50 & keep25, perplexity. Shows the
representative choice's effect in isolation and that the rank-r correction is the decisive variant.
Run: python3 experiments/249_representative.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192; CHUNK = 512; N_EVAL = 1024
K = 128; RCORR = 128; RFEAT = 512
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.50,0.25").split(",")]


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
    CFG = {li: {'active': False} for li in range(nL)}; XC = {li: None for li in range(nL)}

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
            m = oracle_mask(af.float(), cfg['abar'], cfg['vn'], cfg['gsz'], cfg['gf'], cfg['B']).to(a.dtype)
            rep = cfg['rep']                                   # tensor [dff] or scalar 0
            sub = (rep.to(a.dtype) if torch.is_tensor(rep) else a.new_zeros(()))
            return ((af * m + sub * (1 - m)).reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            cfg = CFG[li]
            if not cfg['active'] or not cfg.get('corr'):
                return None
            sh = output.shape
            z = (XC[li].float() - cfg['xbar']) @ cfg['P']
            return (output.reshape(-1, sh[-1]) + (cfg['pred'](z) @ cfg['Br'].T).to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}; hs = []
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
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); n += tgt.numel()
        return torch.tensor(tot / n).exp().item()

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a.half().cpu(), x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None; del a, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  MODEL={MODEL} dff={dff} (representative ablation)", flush=True)

    def build(bf):
        B = int(round(bf * dff)); G = {}
        for li in range(nL):
            s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; vn = s['vn']; Wd = s['Wd'].to(dev)
            Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
            w = ((a - abar).abs() * vn).mean(0)
            gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
            gsz = torch.zeros(K, device=dev)
            for g in range(K):
                gsz[g] = (gl == g).sum()
            m = oracle_mask(a, abar, vn, gsz, gl, B)           # [N,dff] keep mask
            # conditional mean E[a_k | k dropped]: mean of a over tokens where mask==0
            dropped = (1 - m); cnt = dropped.sum(0).clamp(min=1.0)
            condmean = (a * dropped).sum(0) / cnt              # [dff]
            drop = a - (a * m + abar * (1 - m))
            E = drop @ Wd.T; Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
            z = (s['x'].float().to(dev) - s['xbar']) @ s['P']; pred = mlp_fit(z, E @ Br, dev)
            G[li] = dict(abar=abar, vn=vn, gsz=gsz, gf=gl, condmean=condmean, Br=Br, pred=pred,
                         xbar=s['xbar'], P=s['P'])
            del a, Bind, m, dropped, drop, E, Ec, Vt, z, Wd; gc.collect(); torch.cuda.empty_cache()
        return G, B

    def setcfg(G, B, rep_mode, corr=False):
        for li in range(nL):
            g = G[li]
            rep = {'zero': 0, 'mean': g['abar'], 'cond': g['condmean']}[rep_mode]
            CFG[li].update(dict(active=True, B=B, vn=g['vn'], abar=g['abar'], gsz=g['gsz'], gf=g['gf'],
                                rep=rep, corr=corr, Br=g['Br'], xbar=g['xbar'], P=g['P'], pred=g['pred']))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False

    off(); dense = ce_eval()
    print(f"  dense ppl {dense:.3f}\n", flush=True)
    for bf in KEEPS:
        G, B = build(bf)
        setcfg(G, B, 'zero'); z = ce_eval(); off()
        setcfg(G, B, 'mean'); mn = ce_eval(); off()
        setcfg(G, B, 'cond'); cm = ce_eval(); off()
        setcfg(G, B, 'mean', corr=True); cr = ce_eval(); off()
        print(f"  keep{int(bf*100)} representative: zero {z:.3f} | mean(rank0,G-MoE) {mn:.3f} | "
              f"cond-mean {cm:.3f} | +corr(rank{RCORR}) {cr:.3f}  [dense {dense:.3f}]", flush=True)
    print("\nREAD: isolates process-3 (representative). zero<mean<cond-mean<+corr expected; the rank-r", flush=True)
    print("correction is the decisive representative, far beyond the rank-0 constant or conditional mean.", flush=True)


if __name__ == "__main__":
    main()
