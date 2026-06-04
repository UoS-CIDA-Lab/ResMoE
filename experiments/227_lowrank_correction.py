"""Experiment 227 — attack the GROUPING FLOOR (A) via LOW-RANK OUTPUT CORRECTION. Group selection drops
whole groups => error e(t)=sum_{dropped}(a_k-abar_k)v_k in R^d. keep-PATTERN (which neurons) is high-rank
(can't fix grouping), BUT the aggregate OUTPUT error e(t) lives in d-dim and contributions are rank~34
(exp221) => e(t) may be LOW-RANK and correctable WITHOUT fixing the grouping. Test the CEILING: add an
ORACLE rank-r correction e_hat = B_r B_r^T e(t), B_r = top-r left singular vecs of the calib error matrix.
If small r brings group-oracle (4.38) toward dense (3.148)/neu-oracle (3.263) => A is low-rank-correctable
(=> pursue a deployable always-on rank-r path predicted from x). If not => A is high-rank in output too.
Qwen(SwiGLU), keep-pattern grouping K=128, oracle group selection, keep50/25.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/227_lowrank_correction.py
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
RMAX = 128
RANKS = [8, 16, 32, 64, 128]
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
    """group-oracle keep mask [n,dff] (1 kept)."""
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
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    print(f"  MODEL={MODEL} [SwiGLU] layers={nL} K={K} ranks={RANKS}", flush=True)
    torch.set_grad_enabled(False)

    CFG = {li: {'active': False} for li in range(nL)}
    STASH = {li: None for li in range(nL)}

    def dpre(li):
        def hook(_m, args):
            cfg = CFG[li]
            if not cfg['active']:
                return None
            a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1])
            B = cfg['B']; vn = cfg['vn']; abar = cfg['abar']; rep = cfg['rep']
            if cfg.get('neuron'):
                m = keep_topB_neuron(((af.float() - abar).abs() * vn), B).to(a.dtype)
            else:
                m = oracle_mask(af.float(), abar, vn, cfg['gsz'], cfg['gf'], B).to(a.dtype)
            masked = af * m + rep.to(a.dtype) * (1 - m)
            if cfg.get('corr_r'):                            # stash the dropped part for output correction
                STASH[li] = (af - masked).float()            # [n,dff] = (a-rep) on dropped
            return (masked.reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            cfg = CFG[li]
            if not cfg['active'] or not cfg.get('corr_r'):
                return None
            drop = STASH[li]; STASH[li] = None
            Wd = cfg['Wd']; Br = cfg['Br'][:, :cfg['corr_r']]
            e = drop @ Wd.T                                  # [n,d] dropped output error
            ehat = (e @ Br) @ Br.T                           # rank-r projection
            sh = output.shape
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
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

    capa = {li: [] for li in range(nL)}
    hs = [downs[li].register_forward_pre_hook(
        (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li))
        for li in range(nL)]
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"  dff={dff}\n", flush=True)

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li])
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        STR[li] = dict(a=a, abar=abar, vn=vn, Wd=Wd.cpu(), af=None)
        capa[li] = None
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

    def corr_basis(li, bf, gsz, gf):
        """SVD of calib dropped-output-error E[N,d] -> top-RMAX left singular vecs B[d,RMAX]."""
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; vn = s['vn']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        m = oracle_mask(a, abar, vn, gsz, gf, B)
        drop = (a - (a * m + abar * (1 - m)))                # (a-abar) on dropped
        E = drop @ Wd.T                                      # [N,d] dropped output error
        U, S, Vt = torch.linalg.svd(E, full_matrices=False)  # top output directions of the error
        Br = Vt[:RMAX].T                                     # [d, RMAX] (right sing vecs of E = output dirs)
        del a, m, drop, E, U, S, Vt; gc.collect(); torch.cuda.empty_cache()
        return Br

    def setcfg(bf, GRP, mode, r=None, BR=None, neuron=False):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=True, B=B, vn=s['vn'], abar=s['abar'], rep=s['abar'],
                                gsz=gsz, gf=gf, neuron=neuron, corr_r=r,
                                Wd=s['Wd'].to(dev) if r else None, Br=BR[li] if BR else None))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr_r'] = None; CFG[li]['neuron'] = False

    print(f"  SwiGLU ppl (dense {dense_ppl:.3f}), keep-pattern K={K}, oracle group select\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        setcfg(bf, GRP, 'oracle'); base = ppl(); off()
        setcfg(bf, GRP, 'oracle', neuron=True); nu = ppl(); off()
        print(f"  keep{int(bf*100)}:  group-oracle {base:.3f} | neu-oracle {nu:.3f} | dense {dense_ppl:.3f}", flush=True)
        BR = {li: corr_basis(li, bf, *GRP[li]) for li in range(nL)}
        for r in RANKS:
            setcfg(bf, GRP, 'oracle', r=r, BR=BR); v = ppl(); off()
            print(f"           + lowrank-corr r={r:>3} = {v:.3f}", flush=True)
        print("", flush=True)
    print("READ: if +corr brings group-oracle toward dense/neu-oracle at small r => the dropped-output ERROR", flush=True)
    print("is LOW-RANK => grouping floor (A) is correctable by a cheap always-on rank-r path (deployable next).", flush=True)


if __name__ == "__main__":
    main()
