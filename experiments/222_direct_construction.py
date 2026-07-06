"""Experiment 222 — DIRECT construction optimization: is our grouping just suboptimal, or is K=128
fundamentally insufficient? (user: per-neuron works, so expert routing should too with optimal
construction.) The principled construction directly minimizes keep-pattern MIXING: cluster neurons by
their per-token keep-pattern b_k (binary top-B membership), CONTRIBUTION-WEIGHTED (a mixed group hurts
more if trapped neurons have large contributions), balanced. This directly optimizes the block rank-K
approximation of the keep matrix B = the exact thing that determines group-vs-neuron gap.
Compare ORACLE ppl, Qwen(SwiGLU) keep50/25, grp+mean: weight-kmeans (baseline) vs direct keep-pattern
weighted k-means, at K=128 AND K=256, vs neu+cond ceiling. Also report MIXING RATE (fraction of kept
neurons that sit in partially-kept groups) = the interpretable 'tax'.
EXPECT: direct construction may beat weight (construction had room) BUT plateaus far above neu+cond
(rank floor, top-128 keep-energy ~19% from exp221) => most of the gap is the K-expressiveness limit.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/222_direct_construction.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
KS = [128, 256]
KEEPS = [0.50, 0.25]


def weighted_kmeans_centroids(X, w, k, iters=20, seed=0):
    """weighted k-means centroids; X:[n,d] rows=neurons, w:[n] weights."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                wj = w[m]
                c[j] = (X[m] * wj.unsqueeze(1)).sum(0) / wj.sum().clamp(min=1e-6)
    return c


def kmeans_centroids(X, k, iters=15, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
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
    Kc = int(gl.max().item()) + 1
    gsz = torch.zeros(Kc, device=dev); gf = torch.zeros(dff, dtype=torch.long, device=dev)
    for g in range(Kc):
        ix = (gl == g).nonzero().flatten(); gsz[g] = len(ix); gf[ix] = g
    return gsz, gf


def keep_topB_neuron(score, B):
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def keep_topB_group(score_g, gsz, gf, B):
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, gf], selg


def compute_masked(cfg, a):
    vn = cfg['vn']; rep = cfg['rep']; B = cfg['B']
    cost = ((a.float() - rep) * vn) ** 2
    if cfg.get('neuron'):
        keep = keep_topB_neuron(cost, B)
    else:
        Kc = cfg['gsz'].shape[0]
        sg = torch.zeros(a.shape[0], Kc, device=a.device).index_add_(1, cfg['gf'], cost)
        keep, _ = keep_topB_group(sg, cfg['gsz'], cfg['gf'], B)
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
            code += load_dataset("google-research-datasets/mbpp", "full", split=sp)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers
    def parts(l): return l.mlp.down_proj, l.mlp.gate_proj.weight
    nL = len(layers); downs = [parts(layers[li])[0] for li in range(nL)]
    print(f"  MODEL={MODEL} [SwiGLU]  layers={nL}", flush=True)
    torch.set_grad_enabled(False)

    CFG = {li: {'active': False} for li in range(nL)}

    def dpre(li):
        def hook(_m, args):
            cfg = CFG[li]
            if not cfg['active']:
                return None
            a = args[0]; sh = a.shape
            return (compute_masked(cfg, a.reshape(-1, sh[-1])).reshape(sh),) + args[1:]
        return hook
    for li in range(nL):
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
        A = torch.cat(capa[li]); a = A.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wup = parts(layers[li])[1].detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        STR[li] = dict(A=A, mean=abar, vn=vn, Xw=F.normalize(Wup, dim=1).cpu())
        capa[li] = None
        del a, Wd, Wup; gc.collect(); torch.cuda.empty_cache()

    GR = {}; MIX = {}

    def build(li, bf, K, method):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; abar = s['mean']; B = int(round(bf * dff))
        if method == 'weight':
            Xw = s['Xw'].to(dev)
            gl = balanced_assign(Xw, kmeans_centroids(Xw, K, seed=0))
        else:  # direct: contribution-weighted k-means on per-neuron keep-pattern b_k
            Bind = keep_topB_neuron((A - abar).abs() * vn, B).float()    # [N, dff]
            patt = Bind.T.contiguous()                                    # [dff, N] keep-pattern per neuron
            w = ((A - abar).abs() * vn).mean(0)                           # per-neuron mean contribution weight
            c = weighted_kmeans_centroids(patt, w, K, seed=0)
            gl = balanced_assign(patt, c)
        g = grouping_from_assign(gl, dff, dev)
        del A; gc.collect(); torch.cuda.empty_cache()
        return g

    def neu_rep(li, bf):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        keep = keep_topB_neuron((A - s['mean']).abs() * vn, B); dr = (~keep).float()
        rep = (A * dr).sum(0) / dr.sum(0).clamp(min=1.0)
        del A, keep, dr; gc.collect(); torch.cuda.empty_cache()
        return rep

    def set_group(bf, K, method):
        for li in range(nL):
            s = STR[li]; gsz, gf = GR[(li, bf, K, method)]
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), rep=s['mean'], vn=s['vn'],
                                gsz=gsz, gf=gf, neuron=False))

    def set_neuron(bf):
        for li in range(nL):
            s = STR[li]
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), rep=neu_rep(li, bf),
                                vn=s['vn'], neuron=True))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False

    print(f"  SwiGLU ORACLE ppl (dense {dense_ppl:.3f}), grp+mean — weight-kmeans vs DIRECT keep-pattern construction\n", flush=True)
    hdr = "  keep |"
    for K in KS:
        hdr += f" W:K{K}   direct:K{K} |"
    hdr += " neu+cond"
    print(hdr, flush=True)
    for bf in KEEPS:
        for K in KS:
            for method in ['weight', 'direct']:
                for li in range(nL):
                    GR[(li, bf, K, method)] = build(li, bf, K, method)
        row = []
        for K in KS:
            set_group(bf, K, 'weight'); row.append(ppl()); off()
            set_group(bf, K, 'direct'); row.append(ppl()); off()
        set_neuron(bf); nc = ppl(); off()
        cells = "  ".join(f"{v:.3f}" for v in row)
        print(f"  {int(bf*100):>3}% | {cells} | {nc:.3f}", flush=True)
        for K in KS:
            for method in ['weight', 'direct']:
                for li in range(nL):
                    GR.pop((li, bf, K, method), None)
    print("\nREAD: direct < weight => construction had room (user partly right). direct still >> neu+cond", flush=True)
    print("=> the K-expressiveness floor (rank of keep-patterns) dominates; per-neuron unreachable at fixed K.", flush=True)


if __name__ == "__main__":
    main()
