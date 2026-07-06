"""Experiment 220 — grouping blitz: try the UNTRIED levers to close the SwiGLU group→neuron gap at
deployable K=128. Prior methods (weight-kmeans, magnitude/centered-corr kmeans, binary co-occurrence
spectral) all tied/lost. They ignore (i) contribution MAGNITUDE weighting and (ii) OUTPUT-space sign
(cancellation). Here, on Qwen(SwiGLU) keep50/25, ORACLE ppl, grp+mean (best rep for SwiGLU), K=128:
  weight        : kmeans on normalized up-weight rows (baseline)
  spec-bin      : spectral on BINARY top-B co-occurrence (exp219)
  spec-wt       : spectral on CONTRIBUTION-WEIGHTED co-occurrence (Bind*|a|*vn) — high-contrib co-keep
  signed        : kmeans on CENTERED SIGNED contribution profile (a_k-abar)*vn — sign/cancellation
  cancel        : greedy CANCELLATION grouping — pair high-contrib neurons with opposite output dir so
                  group net output is small (=> droppable cheaply); approximate via signed-contrib sign
vs neu+cond ceiling. Any method << weight (toward neu+cond) => deployable breakthrough.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/220_grouping_blitz.py
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
METHODS = ["weight", "spec-bin", "spec-wt", "signed"]


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


def grouping_from_assign(gl, dff, dev):
    Kc = int(gl.max().item()) + 1
    gsz = torch.zeros(Kc, device=dev); gf = torch.zeros(dff, dtype=torch.long, device=dev)
    for g in range(Kc):
        ix = (gl == g).nonzero().flatten(); gsz[g] = len(ix); gf[ix] = g
    return gsz, gf


def keep_topB_neuron(score, B):
    if B >= score.shape[1]:
        return torch.ones_like(score, dtype=torch.bool)
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def keep_topB_group(score_g, gsz, gf, B):
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, gf]


def spectral_emb(M, K):
    """top-K right singular vectors of M [N,dff] (== top eigvecs of co-occurrence MᵀM), NJW-normalized."""
    q = min(K + 16, M.shape[0] - 1, M.shape[1] - 1)
    _, _, V = torch.svd_lowrank(M, q=q)
    return F.normalize(V[:, :K], dim=1)


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
    print(f"  MODEL={MODEL} [{arch}]  layers={nL}  K={KROUTE}", flush=True)
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
    print(f"  dff={dff}", flush=True)

    STR = {}
    for li in range(nL):
        A = torch.cat(capa[li]); a = A.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wup = parts(layers[li])[1].detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        gl_w = balanced_assign(F.normalize(Wup, dim=1), kmeans_centroids(F.normalize(Wup, dim=1), KROUTE, seed=0))
        STR[li] = dict(A=A, mean=abar, vn=vn, grp_w=grouping_from_assign(gl_w, dff, dev))
        capa[li] = None
        del a, Wd, Wup; gc.collect(); torch.cuda.empty_cache()

    def build_group(li, bf, method):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; abar = s['mean']; B = int(round(bf * dff))
        contrib = (A - abar).abs() * vn                       # [N,dff] contribution magnitude
        if method == "weight":
            g = s['grp_w']; del A, contrib; gc.collect(); torch.cuda.empty_cache(); return g
        if method == "spec-bin":
            M = keep_topB_neuron(contrib, B).float()
            gl = balanced_assign(spectral_emb(M, KROUTE), kmeans_centroids(spectral_emb(M, KROUTE), KROUTE, seed=0))
        elif method == "spec-wt":
            M = keep_topB_neuron(contrib, B).float() * contrib   # contribution-weighted co-occurrence
            E = spectral_emb(M, KROUTE)
            gl = balanced_assign(E, kmeans_centroids(E, KROUTE, seed=0))
        else:  # signed: centered SIGNED contribution profile (captures sign/cancellation)
            prof = ((A - abar) * vn).T                          # [dff, N] signed
            prof = prof - prof.mean(1, keepdim=True)
            X = F.normalize(prof, dim=1)
            gl = balanced_assign(X, kmeans_centroids(X, KROUTE, seed=0))
        g = grouping_from_assign(gl, dff, dev)
        del A, contrib; gc.collect(); torch.cuda.empty_cache()
        return g

    def neu_cond(li, bf):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        keep = keep_topB_neuron((A - s['mean']).abs() * vn, B); dr = (~keep).float()
        rep = (A * dr).sum(0) / dr.sum(0).clamp(min=1.0)
        cost = ((A - rep) * vn) ** 2; km = keep_topB_neuron(cost, B).float()
        # neuron oracle ppl is computed via the hooks separately; here just return rep+nothing
        del A, keep, dr, cost, km; gc.collect(); torch.cuda.empty_cache()
        return rep

    GR = {}

    def set_group(bf, method):
        for li in range(nL):
            s = STR[li]; gsz, gf = GR[(li, bf, method)]
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), rep=s['mean'], vn=s['vn'], gsz=gsz, gf=gf))

    def set_neuron(bf):
        for li in range(nL):
            s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
            keep = keep_topB_neuron((A - s['mean']).abs() * vn, B); dr = (~keep).float()
            rep = (A * dr).sum(0) / dr.sum(0).clamp(min=1.0)
            # per-neuron mask via a 1-neuron-per-group trick: store rep, use gran neuron in hook
            CFG[li].update(dict(active=True, B=B, rep=rep, vn=vn, gsz=None, gf=None, neuron=True))
            del A, keep, dr; gc.collect(); torch.cuda.empty_cache()

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['neuron'] = False

    print(f"\n  {arch} ORACLE ppl (dense {dense_ppl:.3f}), grp+mean, K={KROUTE}\n", flush=True)
    print("  keep |" + "".join(f" {m:>9}" for m in METHODS) + " | neu+cond", flush=True)
    for bf in KEEPS:
        for method in METHODS:
            for li in range(nL):
                GR[(li, bf, method)] = build_group(li, bf, method)
        row = []
        for method in METHODS:
            set_group(bf, method); row.append(ppl()); off()
        set_neuron(bf); nc = ppl(); off()
        print(f"  {int(bf*100):>3}% |" + "".join(f" {v:>9.3f}" for v in row) + f" | {nc:.3f}", flush=True)
        for method in METHODS:
            for li in range(nL):
                GR.pop((li, bf, method), None)
    print("\nREAD: any method << weight (toward neu+cond) => deployable breakthrough at K=128. If all ≈ weight", flush=True)
    print("=> contribution-weighting and output-sign don't expose fixed-K structure either.", flush=True)


if __name__ == "__main__":
    main()
