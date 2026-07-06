"""Experiment 215 — does the CONDITIONAL-MEAN representative help on GeLU & SwiGLU (our targets)?
exp214 proved on OPT(ReLU) that conditional mean E[a_k|dropped] >> unconditional mean (G-MoE's). The
dropped set is a low-activation biased subsample for ANY activation, so the bias-correction should help
GeLU/SwiGLU too (less dramatically than ReLU, since their mean is already a decent estimate). Test
ORACLE ppl, route-all (sf=0), keep50/25: grp+mean (=G-MoE) vs grp+cond vs neu+cond.
Arch-agnostic: mask injected at the DOWN-projection pre-hook (its input is the post-activation hidden a,
true for both SwiGLU down_proj and GeLU c_proj). Run:
  HHMODEL=bigcode/gpt_bigcode-santacoder python3 experiments/215_condmean_gelu_swiglu.py
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B    python3 experiments/215_condmean_gelu_swiglu.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "bigcode/gpt_bigcode-santacoder")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
KROUTE = 64
KEEPS = [0.50, 0.25]


def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a


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
    # arch detection: locate decoder layers, the down-projection, and the up-weight for grouping
    if hasattr(model, "model") and hasattr(model.model, "layers"):        # Qwen2 / LLaMA SwiGLU
        layers = model.model.layers
        def parts(l): return l.mlp.down_proj, l.mlp.gate_proj.weight
        arch = "SwiGLU"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT2/SantaCoder GeLU
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
    print(f"  dff={dff}", flush=True)

    STR = {}
    for li in range(nL):
        A = torch.cat(capa[li])
        a = A.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wup = parts(layers[li])[1].detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        vn = Wd.norm(dim=0)
        grp_local = kmeans(F.normalize(Wup, dim=1), KROUTE, seed=0)
        Kc = int(grp_local.max().item()) + 1
        gsz = torch.zeros(Kc, device=dev); grp_full = torch.zeros(dff, dtype=torch.long, device=dev)
        for g in range(Kc):
            ix = (grp_local == g).nonzero().flatten()
            gsz[g] = len(ix); grp_full[ix] = g
        STR[li] = dict(A=A, mean=a.mean(0), vn=vn, gsz=gsz, grp_full=grp_full)
        capa[li] = None
        del a, Wd, Wup; gc.collect(); torch.cuda.empty_cache()

    def condmean(li, bf, gran):
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        score0 = (A.abs() * vn)
        if gran == 'neuron':
            keep = keep_topB_neuron(score0, B)
        else:
            Kc = s['gsz'].shape[0]
            sg = torch.zeros(A.shape[0], Kc, device=dev).index_add_(1, s['grp_full'], score0 ** 2)
            keep = keep_topB_group(sg, s['gsz'], s['grp_full'], B)
        dropped = (~keep).float()
        rep = (A * dropped).sum(0) / dropped.sum(0).clamp(min=1.0)
        del A, score0, keep, dropped; gc.collect(); torch.cuda.empty_cache()
        return rep

    def setcfg(bf, gran, reptype):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]
            rep = s['mean'] if reptype == 'mean' else condmean(li, bf, gran)
            CFG[li].update(dict(active=True, B=B, gran=gran, rep=rep, vn=s['vn'],
                                gsz=s['gsz'], grp_full=s['grp_full']))

    def set_active(flag):
        for li in range(nL):
            CFG[li]['active'] = flag

    print(f"\n  {arch} ORACLE ppl (dense {dense_ppl:.3f}): unconditional-mean(G-MoE) vs conditional-mean\n", flush=True)
    print(f"  {'keep':>5} | {'grp+mean':>9} | {'grp+cond':>9} | {'neu+cond':>9}", flush=True)
    for bf in KEEPS:
        r = {}
        for gran, reptype, key in [('group', 'mean', 'gm'), ('group', 'cond', 'gc'), ('neuron', 'cond', 'nc')]:
            setcfg(bf, gran, reptype); r[key] = ppl(); set_active(False)
        print(f"  {int(bf*100):>4}% | {r['gm']:>9.3f} | {r['gc']:>9.3f} | {r['nc']:>9.3f}", flush=True)
    print(f"\nREAD [{arch}]: grp+cond < grp+mean => conditional-mean rep helps beyond ReLU (general", flush=True)
    print("improvement over G-MoE's unconditional mean). Smaller gain than ReLU expected (mean already ok).", flush=True)


if __name__ == "__main__":
    main()
