"""Experiment 217 — BALANCED grouping: fair test of co-activation vs weight (exp216 was confounded by
imbalance — naive co-act kmeans made giant groups => C:grp+mean blew up to 2352 on Qwen keep25). Here
BOTH groupings use balanced assignment (equal group sizes, like MoEfication/G-MoE), isolating the
grouping CRITERION (weight-direction vs co-activation-profile). Compare ORACLE ppl keep50/25, route-all:
balanced-weight {grp+mean, grp+cond} vs balanced-coact {grp+mean, grp+cond} vs neu+cond ceiling.
QUESTION (user's): does co-activation grouping reduce the active/inactive MIXING in dropped groups so
deployable (structured) group selection approaches the per-neuron ceiling? Arch-agnostic.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/217_balanced_grouping.py
     HHMODEL=bigcode/gpt_bigcode-santacoder python3 experiments/217_balanced_grouping.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
KROUTE = 64
KEEPS = [0.50, 0.25]
COACT_DIMS = 4096      # calib tokens used for the (centered/correlation) co-activation profile


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
    """Equal-size assignment: capacity ceil(n/k); greedy by regret (most decisive first)."""
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
            code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        def parts(l): return l.mlp.down_proj, l.mlp.gate_proj.weight
        arch = "SwiGLU"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        def parts(l): return l.mlp.c_proj, l.mlp.c_fc.weight
        arch = "GeLU"
    else:
        raise RuntimeError("unknown arch")
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
    print(f"  dff={dff}; building BALANCED weight & co-activation groupings", flush=True)

    STR = {}
    for li in range(nL):
        A = torch.cat(capa[li]); a = A.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wup = parts(layers[li])[1].detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        vn = Wd.norm(dim=0)
        Xw = F.normalize(Wup, dim=1)
        # PROPER co-activation: CENTER each neuron's profile (per-token mean removed) so cosine == Pearson
        # correlation => groups neurons that fire TOGETHER (co-deviation pattern), not by magnitude.
        prof = a[:COACT_DIMS].T                                   # [dff, T]
        prof = prof - prof.mean(1, keepdim=True)
        Xc = F.normalize(prof, dim=1)
        gl_w = balanced_assign(Xw, kmeans_centroids(Xw, KROUTE, seed=0))
        gl_c = balanced_assign(Xc, kmeans_centroids(Xc, KROUTE, seed=0))
        gsz_w, gf_w = grouping_from_assign(gl_w, dff, dev)
        gsz_c, gf_c = grouping_from_assign(gl_c, dff, dev)
        STR[li] = dict(A=A, mean=a.mean(0), vn=vn, gsz_w=gsz_w, gf_w=gf_w, gsz_c=gsz_c, gf_c=gf_c)
        capa[li] = None
        del a, Wd, Wup, Xw, Xc; gc.collect(); torch.cuda.empty_cache()
    # report balance achieved
    print(f"  group sizes: weight[min={int(STR[0]['gsz_w'].min())},max={int(STR[0]['gsz_w'].max())}] "
          f"coact[min={int(STR[0]['gsz_c'].min())},max={int(STR[0]['gsz_c'].max())}] (dff/K={dff//KROUTE})", flush=True)

    def condmean(li, bf, gran, gf, gsz):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        score0 = (A.abs() * vn)
        if gran == 'neuron':
            keep = keep_topB_neuron(score0, B)
        else:
            Kc = gsz.shape[0]
            sg = torch.zeros(A.shape[0], Kc, device=dev).index_add_(1, gf, score0 ** 2)
            keep = keep_topB_group(sg, gsz, gf, B)
        dropped = (~keep).float()
        rep = (A * dropped).sum(0) / dropped.sum(0).clamp(min=1.0)
        del A, score0, keep, dropped; gc.collect(); torch.cuda.empty_cache()
        return rep

    def setcfg(bf, gran, reptype, gmode):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]
            gf = s['gf_c'] if gmode == 'coact' else s['gf_w']
            gsz = s['gsz_c'] if gmode == 'coact' else s['gsz_w']
            rep = s['mean'] if reptype == 'mean' else condmean(li, bf, gran, gf, gsz)
            CFG[li].update(dict(active=True, B=B, gran=gran, rep=rep, vn=s['vn'], gsz=gsz, grp_full=gf))

    def set_active(flag):
        for li in range(nL):
            CFG[li]['active'] = flag

    print(f"\n  {arch} ORACLE ppl (dense {dense_ppl:.3f}): BALANCED weight vs BALANCED co-activation\n", flush=True)
    print(f"  {'keep':>5} | {'W:grp+mean':>11} {'W:grp+cond':>11} | {'C:grp+mean':>11} {'C:grp+cond':>11} | {'neu+cond':>9}", flush=True)
    for bf in KEEPS:
        r = {}
        for gm in ['weight', 'coact']:
            setcfg(bf, 'group', 'mean', gm); r[(gm, 'm')] = ppl(); set_active(False)
            setcfg(bf, 'group', 'cond', gm); r[(gm, 'c')] = ppl(); set_active(False)
        setcfg(bf, 'neuron', 'cond', 'weight'); nc = ppl(); set_active(False)
        print(f"  {int(bf*100):>4}% | {r[('weight','m')]:>11.3f} {r[('weight','c')]:>11.3f} | "
              f"{r[('coact','m')]:>11.3f} {r[('coact','c')]:>11.3f} | {nc:>9.3f}", flush=True)
    print(f"\nREAD [{arch}]: balanced removes the size confound. If C < W now => co-activation grouping", flush=True)
    print("genuinely reduces mixing (deployable path toward neu+cond). If C≈W or worse => the grouping", flush=True)
    print("criterion doesn't help; structured-selection floor is robust (per-neuron ceiling unreachable deployably).", flush=True)


if __name__ == "__main__":
    main()
