"""Experiment 231 — EQUAL-COST: is the low-rank correction (+3% FFN) better than just keeping +3% more
neurons? The r=128 correction costs corr_frac=(hidden*rfeat + rfeat*h + h*r + r*hidden)/(3*hidden*dff)
≈3.17% FFN. So 'keep f + correction' ≈ 'keep f+corr_frac (no correction)' in compute. Compare at equal
compute (ORACLE selection for both, to isolate correction-vs-more-neurons):
  group-oracle(f)                         [compute f]
  group-oracle(f) + predicted-corr r128   [compute f+corr_frac]   <- correction
  group-oracle(f+corr_frac)               [compute f+corr_frac]   <- more neurons (equal cost)
  refs neu-oracle(f), dense.
If (f + corr) < (f+corr_frac no-corr) => correction is a BETTER use of the budget than more neurons.
Qwen(SwiGLU) keep50/25. Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/231_corr_vs_moreneurons.py
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
RCORR = 128
RFEAT = 512
HID = 512
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


def mlp_fit(X, Y, dev, steps=3000, hidden=HID, lr=3e-3, bs=2048):
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
            x = XC[li].float(); sh = output.shape
            z = (x - cfg['xbar']) @ cfg['P']
            ehat = cfg['pred'](z) @ cfg['Br'].T
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
    print(f"  MODEL={MODEL} dff={dff} hidden={hidden} | corr_frac(r{RCORR})={corr_frac*100:.2f}% FFN | dense {dense_ppl:.3f}\n", flush=True)

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), Wd=Wd.cpu(), abar=abar, vn=vn, xbar=xbar, P=VtX[:RFEAT].T)
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

    def train_corr(li, bf, gsz, gf):
        s = STR[li]; a = s['a'].float().to(dev); Wd = s['Wd'].to(dev); x = s['x'].float().to(dev)
        B = int(round(bf * dff))
        m = oracle_mask(a, s['abar'], s['vn'], gsz, gf, B)
        drop = a - (a * m + s['abar'] * (1 - m))
        E = drop @ Wd.T; _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
        c = E @ Br; z = (x - s['xbar']) @ s['P']; pred = mlp_fit(z, c, dev)
        del a, Wd, x, m, drop, E, Vt, c, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def run(GRP, bf, corr=False, BR=None, PRED=None, neuron=False):
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), abar=s['abar'], vn=s['vn'],
                                gsz=gsz, gf=gf, neuron=neuron, corr=corr, xbar=s['xbar'], P=s['P'],
                                Br=BR[li] if BR else None, pred=PRED[li] if PRED else None))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    print(f"  keep | group-oracle(f) | +corr r{RCORR} | more-neurons(f+{corr_frac*100:.1f}%) | neu-oracle | dense", flush=True)
    for bf in KEEPS:
        bf2 = bf + corr_frac
        GRP = {li: grouping(li, bf) for li in range(nL)}
        GRP2 = {li: grouping(li, bf2) for li in range(nL)}
        run(GRP, bf); base = ppl(); off()
        run(GRP, bf, neuron=True); nu = ppl(); off()
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = train_corr(li, bf, *GRP[li])
        run(GRP, bf, corr=True, BR=BR, PRED=PRED); corr = ppl(); off()
        run(GRP2, bf2); more = ppl(); off()
        win = "CORR wins" if corr < more else "MORE-NEURONS wins"
        print(f"  {int(bf*100):>3}% | {base:>14.3f} | {corr:>10.3f} | {more:>20.3f} | {nu:>10.3f} | {dense_ppl:.3f}  -> {win}", flush=True)
    print("\nREAD: +corr vs more-neurons at EQUAL compute. CORR wins => low-rank correction is a better use", flush=True)
    print("of the budget than keeping more neurons (unlike static low-rank rep, which lost to more neurons).", flush=True)


if __name__ == "__main__":
    main()
