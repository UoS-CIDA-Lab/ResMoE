"""Experiment 223 — SELECTION-AWARE LOCAL SEARCH: squeeze construction beyond direct keep-pattern (222).
Clustering ignores (a) output-space CANCELLATION and (b) the top-group SELECTION nonlinearity. Local
search optimizes the TRUE oracle reconstruction error in a low-rank OUTPUT space (contrib stable-rank
≈34, exp221): refine the grouping by batched neuron SWAPS that reduce E=sum_t sum_{dropped g}||rho_g(t)||^2
(fixed-selection delta), balance preserved. Init = direct keep-pattern construction.
Compare ORACLE ppl (grp+mean), Qwen(SwiGLU) keep50/25, K=128: direct(init) vs +localsearch vs neu+cond.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/223_local_search.py
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
KEEPS = [0.50, 0.25]
T_LS = 1024
R_LS = 48
ROUNDS = 400
BATCH = 4096


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
    Kc = K
    gsz = torch.zeros(Kc, device=dev); gf = gl.clone()
    for g in range(Kc):
        gsz[g] = (gl == g).sum()
    return gsz, gf


def keep_topB_neuron(score, B):
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def keep_topB_group(score_g, gsz, gf, B):
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, gf]


def compute_masked(cfg, a):
    vn = cfg['vn']; rep = cfg['rep']; B = cfg['B']
    cost = ((a.float() - rep) * vn) ** 2
    if cfg.get('neuron'):
        keep = keep_topB_neuron(cost, B)
    else:
        Kc = cfg['gsz'].shape[0]
        sg = torch.zeros(a.shape[0], Kc, device=a.device).index_add_(1, cfg['gf'], cost)
        keep = keep_topB_group(sg, cfg['gsz'], cfg['gf'], B)
    m = keep.to(a.dtype)
    return a * m + rep.to(a.dtype) * (1 - m)


def local_search(gl, rc, B, dev, rounds=ROUNDS, batch=BATCH, seed=0):
    """gl:[dff] init. rc:[T,dff,r] reduced per-token contributions. balanced swaps minimizing dropped E."""
    dff = gl.shape[0]; Tn = rc.shape[0]
    gen = torch.Generator(device=dev).manual_seed(seed)
    onehot = F.one_hot(gl, K).float()
    rho = torch.einsum('dk,tdr->tkr', onehot, rc)          # [T,K,r]
    gsz = onehot.sum(0)                                     # [K] (constant under swaps)

    def dropped_mask():
        nrm = (rho * rho).sum(-1)                           # [T,K]
        order = nrm.argsort(1, descending=True); so = gsz[order]
        keep_ord = (so.cumsum(1) - so) < B
        kept = torch.zeros(Tn, K, dtype=torch.bool, device=dev).scatter_(1, order, keep_ord)
        return (~kept).float()

    def energy(D):
        return (D * (rho * rho).sum(-1)).sum().item()

    D = dropped_mask(); E0 = energy(D)
    for rd in range(rounds):
        i = torch.randint(0, dff, (batch,), generator=gen, device=dev)
        j = torch.randint(0, dff, (batch,), generator=gen, device=dev)
        gi = gl[i]; gj = gl[j]; ok = gi != gj
        i, j, gi, gj = i[ok], j[ok], gi[ok], gj[ok]
        if i.numel() == 0:
            continue
        delta = rc[:, j, :] - rc[:, i, :]                   # [T,b,r]; gi gains j loses i => +delta
        di = (rho[:, gi, :] * delta).sum(-1)                # [T,b]
        dj = (rho[:, gj, :] * delta).sum(-1)
        d2 = (delta * delta).sum(-1)
        dE = (D[:, gi] * (2 * di + d2) + D[:, gj] * (-2 * dj + d2)).sum(0)   # [b]
        cand = (dE < -1e-6).nonzero().flatten()
        if cand.numel() == 0:
            continue
        cand = cand[dE[cand].argsort()]
        ii = i.tolist(); jj = j.tolist(); gii = gi.tolist(); gjj = gj.tolist()
        used_n = set(); used_g = set(); appl = []
        for c in cand.tolist():
            a, b, ga, gb = ii[c], jj[c], gii[c], gjj[c]
            if a in used_n or b in used_n or ga in used_g or gb in used_g:
                continue
            used_n |= {a, b}; used_g |= {ga, gb}; appl.append(c)
        if not appl:
            continue
        appl = torch.tensor(appl, device=dev)
        ia, ja, gia, gja = i[appl], j[appl], gi[appl], gj[appl]
        dl = rc[:, ja, :] - rc[:, ia, :]
        rho[:, gia, :] += dl
        rho[:, gja, :] -= dl
        gl[ia] = gja; gl[ja] = gia
        if rd % 20 == 19:
            D = dropped_mask()
    return gl, E0, energy(dropped_mask())


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
    print(f"  MODEL={MODEL} [SwiGLU] layers={nL} K={K} (LS T={T_LS} r={R_LS} rounds={ROUNDS})", flush=True)
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
        STR[li] = dict(A=A, mean=a.mean(0), vn=Wd.norm(dim=0), Wd=Wd.cpu())
        capa[li] = None
        del a, Wd; gc.collect(); torch.cuda.empty_cache()

    def build_direct(li, bf):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; abar = s['mean']; B = int(round(bf * dff))
        Bind = keep_topB_neuron((A - abar).abs() * vn, B).float()
        patt = Bind.T.contiguous(); w = ((A - abar).abs() * vn).mean(0)
        gl = balanced_assign(patt, weighted_kmeans_centroids(patt, w, K, seed=0))
        del A, Bind, patt; gc.collect(); torch.cuda.empty_cache()
        return gl

    def reduced_contrib(li):
        s = STR[li]; A = s['A'][:T_LS].float().to(dev); abar = s['mean']; Wd = s['Wd'].to(dev)
        r = (A - abar)                                       # [T,dff]
        U, _, _ = torch.svd_lowrank(Wd, q=R_LS)              # U:[d,R] top output dirs
        v_proj = Wd.T @ U                                    # [dff,R]
        rc = r.unsqueeze(2) * v_proj.unsqueeze(0)            # [T,dff,R]
        del A, Wd, U, v_proj, r; gc.collect(); torch.cuda.empty_cache()
        return rc

    def set_group(gls, bf):
        for li in range(nL):
            s = STR[li]; gsz, gf = grouping_from_assign(gls[li], dff, dev)
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), rep=s['mean'], vn=s['vn'],
                                gsz=gsz, gf=gf, neuron=False))

    def set_neuron(bf):
        for li in range(nL):
            s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
            keep = keep_topB_neuron((A - s['mean']).abs() * vn, B); dr = (~keep).float()
            rep = (A * dr).sum(0) / dr.sum(0).clamp(min=1.0)
            CFG[li].update(dict(active=True, B=B, rep=rep, vn=vn, neuron=True))
            del A, keep, dr; gc.collect(); torch.cuda.empty_cache()

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['neuron'] = False

    print(f"  SwiGLU ORACLE ppl (dense {dense_ppl:.3f}), grp+mean, K={K}\n", flush=True)
    print(f"  {'keep':>5} | {'direct':>8} | {'+LS':>8} | {'neu+cond':>9}", flush=True)
    for bf in KEEPS:
        gl_init = {li: build_direct(li, bf) for li in range(nL)}
        set_group(gl_init, bf); d0 = ppl(); off()
        gl_ls = {}; deltas = []
        for li in range(nL):
            rc = reduced_contrib(li)
            gl, e0, e1 = local_search(gl_init[li].clone(), rc, int(round(bf * dff)), dev)
            gl_ls[li] = gl; deltas.append((e0 - e1) / max(e0, 1e-9))
            del rc; gc.collect(); torch.cuda.empty_cache()
        set_group(gl_ls, bf); dls = ppl(); off()
        set_neuron(bf); nc = ppl(); off()
        print(f"  {int(bf*100):>4}% | {d0:>8.3f} | {dls:>8.3f} | {nc:>9.3f}   (LS reduced-E drop {sum(deltas)/len(deltas)*100:.1f}%)", flush=True)
    print("\nREAD: +LS << direct => more deployable construction juice. +LS≈direct => near the fixed-K", flush=True)
    print("construction ceiling; residual to neu+cond = rank floor (unreachable by any K=128 grouping).", flush=True)


if __name__ == "__main__":
    main()
