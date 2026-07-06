"""Experiment 219 — CO-OCCURRENCE SPECTRAL grouping vs weight-kmeans, fixed K=128 (user: a better
fixed-K algorithm should exist). Current weight-kmeans groups by INPUT-WEIGHT direction (proxy; blind to
actual activations, output v_k, and the selection objective). Here we group by ACTUAL co-firing:
  - binary indicator b_k(t)=1 if neuron k is in token t's top-B contributor set (|a_k|*||v_k||) => [N,dff]
  - co-occurrence affinity A=Bind^T Bind; spectral embedding = top-K right singular vectors of Bind
    (== top-K eigvecs of A, via svd_lowrank, avoiding a dff×dff eigh), NJW row-normalize, balanced k-means.
This groups neurons that are JOINTLY KEPT across tokens => dropped groups should be cleanly co-inactive.
Compare ORACLE ppl, Qwen(SwiGLU) keep50/25, K=128: weight {grp+mean,grp+cond} vs spectral {grp+mean,
grp+cond} vs neu+cond ceiling. spectral is keep-specific (uses top-B); weight is keep-agnostic.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/219_spectral_grouping.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
KROUTE = 128
KEEPS = [0.50, 0.25]


def kmeans_centroids(X, k, iters=12, seed=0):
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


def grouping_from_assign(grp_local, dff, dev):
    Kc = int(grp_local.max().item()) + 1
    gsz = torch.zeros(Kc, device=dev); grp_full = torch.zeros(dff, dtype=torch.long, device=dev)
    for g in range(Kc):
        ix = (grp_local == g).nonzero().flatten()
        gsz[g] = len(ix); grp_full[ix] = g
    return gsz, grp_full


def keep_topB_neuron(score, B):
    if B >= score.shape[1]:
        return torch.ones_like(score, dtype=torch.bool)
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def keep_topB_group(score_g, gsz, grp_full, B):
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, grp_full]


def spectral_group(Bind, K, dev):
    """Bind:[N,dff] binary co-firing indicator. Spectral embedding = top-K right singular vecs."""
    q = min(K + 16, Bind.shape[0] - 1, Bind.shape[1] - 1)
    _, _, V = torch.svd_lowrank(Bind, q=q)        # V:[dff,q] = top eigvecs of Bind^T Bind (co-occurrence)
    emb = F.normalize(V[:, :K], dim=1)            # NJW row-normalize
    return balanced_assign(emb, kmeans_centroids(emb, K, seed=0))


def compute_masked(cfg, a):
    vn = cfg['vn']; rep = cfg['rep']; B = cfg['B']
    cost = ((a.float() - rep) * vn) ** 2
    if cfg['gran'] == 'neuron':
        keep = keep_topB_neuron(cost, B)
    else:
        Kc = cfg['gsz'].shape[0]
        sg = torch.zeros(a.shape[0], Kc, device=a.device).index_add_(1, cfg['grp_full'], cost)
        keep = keep_topB_group(sg, cfg['gsz'], cfg['grp_full'], B)
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
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        def parts(l): return l.mlp.down_proj, l.mlp.gate_proj.weight
        arch = "SwiGLU"
    else:
        layers = model.transformer.h
        def parts(l): return l.mlp.c_proj, l.mlp.c_fc.weight
        arch = "GeLU"
    nL = len(layers); downs = [parts(layers[li])[0] for li in range(nL)]
    print(f"  MODEL={MODEL} [{arch}]  tokens={ids.numel()}  layers={nL}  K={KROUTE}", flush=True)
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
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
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
    print(f"  dff={dff}; building weight grouping (keep-agnostic)", flush=True)

    STR = {}
    for li in range(nL):
        A = torch.cat(capa[li]); a = A.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wup = parts(layers[li])[1].detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        vn = Wd.norm(dim=0); Xw = F.normalize(Wup, dim=1)
        gl_w = balanced_assign(Xw, kmeans_centroids(Xw, KROUTE, seed=0))
        STR[li] = dict(A=A, mean=a.mean(0), vn=vn, grp_w=grouping_from_assign(gl_w, dff, dev))
        capa[li] = None
        del a, Wd, Wup, Xw; gc.collect(); torch.cuda.empty_cache()

    def build_spectral(li, bf):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        Bind = keep_topB_neuron(A.abs() * vn, B).float()        # [N,dff] co-firing indicator @ this keep
        gl = spectral_group(Bind, KROUTE, dev)
        g = grouping_from_assign(gl, dff, dev)
        del A, Bind; gc.collect(); torch.cuda.empty_cache()
        return g

    def condmean(li, bf, gf, gsz):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        score0 = (A.abs() * vn)
        Kc = gsz.shape[0]
        sg = torch.zeros(A.shape[0], Kc, device=dev).index_add_(1, gf, score0 ** 2)
        keep = keep_topB_group(sg, gsz, gf, B); dropped = (~keep).float()
        rep = (A * dropped).sum(0) / dropped.sum(0).clamp(min=1.0)
        del A, score0, keep, dropped; gc.collect(); torch.cuda.empty_cache()
        return rep

    def condmean_neuron(li, bf):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        keep = keep_topB_neuron(A.abs() * vn, B); dropped = (~keep).float()
        rep = (A * dropped).sum(0) / dropped.sum(0).clamp(min=1.0)
        del A, keep, dropped; gc.collect(); torch.cuda.empty_cache()
        return rep

    SPEC = {}  # cache spectral groupings per keep

    def setcfg(bf, gmode, reptype):
        for li in range(nL):
            s = STR[li]
            if gmode == 'weight':
                gsz, gf = s['grp_w']
            else:
                gsz, gf = SPEC[(li, bf)]
            rep = s['mean'] if reptype == 'mean' else condmean(li, bf, gf, gsz)
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), gran='group', rep=rep,
                                vn=s['vn'], gsz=gsz, grp_full=gf))

    def set_neuron(bf):
        for li in range(nL):
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), gran='neuron',
                                rep=condmean_neuron(li, bf), vn=STR[li]['vn']))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False

    print(f"\n  {arch} ORACLE ppl (dense {dense_ppl:.3f}), K={KROUTE}: WEIGHT-kmeans vs CO-OCCURRENCE SPECTRAL\n", flush=True)
    print(f"  {'keep':>5} | {'W:mean':>8} {'W:cond':>8} | {'S:mean':>8} {'S:cond':>8} | {'neu+cond':>9}", flush=True)
    for bf in KEEPS:
        for li in range(nL):
            SPEC[(li, bf)] = build_spectral(li, bf)
        r = {}
        setcfg(bf, 'weight', 'mean'); r['wm'] = ppl(); off()
        setcfg(bf, 'weight', 'cond'); r['wc'] = ppl(); off()
        setcfg(bf, 'spectral', 'mean'); r['sm'] = ppl(); off()
        setcfg(bf, 'spectral', 'cond'); r['sc'] = ppl(); off()
        set_neuron(bf); nc = ppl(); off()
        print(f"  {int(bf*100):>4}% | {r['wm']:>8.3f} {r['wc']:>8.3f} | {r['sm']:>8.3f} {r['sc']:>8.3f} | {nc:>9.3f}", flush=True)
        for li in range(nL):
            SPEC.pop((li, bf), None)
    print(f"\nREAD [{arch}]: if S < W (toward neu+cond) => co-occurrence spectral grouping beats weight-kmeans", flush=True)
    print("at fixed deployable K=128 (user right: a better fixed-K algorithm exists). If S≈W => the activation", flush=True)
    print("co-firing has no fixed-K block structure spectral can exploit beyond weight directions.", flush=True)


if __name__ == "__main__":
    main()
