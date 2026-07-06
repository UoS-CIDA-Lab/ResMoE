"""Experiment 218 — GROUP-COUNT (K) sweep: is the group→neuron gap a coarse-grouping problem (fixable
by 'doing grouping better/finer') or a fundamental structured-selection limit? (user's pushback on 217.)
As K grows (groups shrink), group selection gains finer control => mixing falls => group-oracle should
approach the per-neuron ceiling neu+cond; at K=dff it IS per-neuron. The RATE answers it:
  - if a DEPLOYABLE K (say 128-512) already ≈ neu+cond => better/finer grouping is the answer (user right),
    and that K is the deployable sweet spot.
  - if it only converges near K≈dff => effectively per-neuron (no structured speedup) => structural limit.
Qwen(SwiGLU, the hard case) keep50/25, BALANCED weight grouping, grp+mean & grp+cond vs neu+cond.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/218_ksweep_grouping.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
KS = [64, 128, 256, 512, 1024]
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
    print(f"  MODEL={MODEL} [{arch}]  tokens={ids.numel()}  layers={nL}", flush=True)
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
    print(f"  dff={dff}; precomputing weight groupings for K={KS}", flush=True)

    # precompute balanced weight grouping for each K, per layer
    STR = {}
    for li in range(nL):
        A = torch.cat(capa[li]); a = A.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wup = parts(layers[li])[1].detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        vn = Wd.norm(dim=0); Xw = F.normalize(Wup, dim=1)
        groups = {}
        for K in KS:
            gl = balanced_assign(Xw, kmeans_centroids(Xw, K, seed=0))
            groups[K] = grouping_from_assign(gl, dff, dev)
        STR[li] = dict(A=A, mean=a.mean(0), vn=vn, groups=groups)
        capa[li] = None
        del a, Wd, Wup, Xw; gc.collect(); torch.cuda.empty_cache()

    def condmean_group(li, bf, gf, gsz):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        score0 = (A.abs() * vn)
        Kc = gsz.shape[0]
        sg = torch.zeros(A.shape[0], Kc, device=dev).index_add_(1, gf, score0 ** 2)
        keep = keep_topB_group(sg, gsz, gf, B)
        dropped = (~keep).float()
        rep = (A * dropped).sum(0) / dropped.sum(0).clamp(min=1.0)
        del A, score0, keep, dropped; gc.collect(); torch.cuda.empty_cache()
        return rep

    def condmean_neuron(li, bf):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        keep = keep_topB_neuron(A.abs() * vn, B); dropped = (~keep).float()
        rep = (A * dropped).sum(0) / dropped.sum(0).clamp(min=1.0)
        del A, keep, dropped; gc.collect(); torch.cuda.empty_cache()
        return rep

    def set_group(bf, K, reptype):
        for li in range(nL):
            s = STR[li]; gsz, gf = s['groups'][K]
            rep = s['mean'] if reptype == 'mean' else condmean_group(li, bf, gf, gsz)
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), gran='group', rep=rep,
                                vn=s['vn'], gsz=gsz, grp_full=gf))

    def set_neuron(bf):
        for li in range(nL):
            s = STR[li]
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), gran='neuron',
                                rep=condmean_neuron(li, bf), vn=s['vn']))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False

    print(f"\n  {arch} ORACLE ppl (dense {dense_ppl:.3f}) — group-count K sweep, balanced weight grouping\n", flush=True)
    hdr = "  keep |" + "".join(f" K={K:<5}" for K in KS) + " | neuron"
    print(hdr, flush=True)
    for bf in KEEPS:
        # grp+cond across K
        row_c = []
        for K in KS:
            set_group(bf, K, 'cond'); row_c.append(ppl()); off()
        set_neuron(bf); nc = ppl(); off()
        print(f"  {int(bf*100):>3}%c|" + "".join(f" {v:<7.3f}" for v in row_c) + f" | {nc:.3f}", flush=True)
        # grp+mean across K
        row_m = []
        for K in KS:
            set_group(bf, K, 'mean'); row_m.append(ppl()); off()
        print(f"  {int(bf*100):>3}%m|" + "".join(f" {v:<7.3f}" for v in row_m) + f" |", flush=True)
    print("\nREAD: 'c'=grp+cond, 'm'=grp+mean. If ppl drops toward 'neuron' at a DEPLOYABLE K (128-512) =>", flush=True)
    print("finer/better grouping closes the gap (user right; that K = sweet spot). If it only converges near", flush=True)
    print("K≈dff => effectively per-neuron (no structured speedup) => structural limit is real.", flush=True)


if __name__ == "__main__":
    main()
