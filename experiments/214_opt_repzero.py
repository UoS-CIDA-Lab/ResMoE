"""Experiment 214 — CONDITIONAL-MEAN representative: fix the mean-rep blind spot (user's insight).
The representative for a DROPPED neuron should be E[a_k | k is dropped] (mean activation CONDITIONED
on being selected for dropping), NOT the unconditional mean abar_k. The dropped set is a biased
(low-activation) subsample, so unconditional mean OVER-estimates dropped neurons => injects bias. For
ReLU, dropped = inactive => E[a_k|dropped] ≈ 0 (so 'rep=0 for ReLU' is the special case of this general
principle). G-MoE uses unconditional mean = blind spot. This experiment tests, on OPT(ReLU), ORACLE ppl:
  grp_mean     : group select, rep = UNCONDITIONAL mean   [= G-MoE / exp213 baseline, collapses]
  grp_condmean : group select, rep = CONDITIONAL mean E[a_k|group dropped]
  neuron_condmean : per-neuron select, rep = CONDITIONAL mean E[a_k|neuron dropped]
  neuron_zero  : per-neuron select, rep = 0  (ReLU limiting case; sanity vs condmean)
dropped-set for the conditional mean is defined on calibration by the rep-independent rule 'keep top-B
by |a_k|*||v_k||'. PREDICTION: condmean ≈ neuron_zero ≈ dense at aggressive keep => collapse was the
mean-rep, not ReLU. Run: HHMODEL=facebook/opt-1.3b python3 experiments/214_opt_repzero.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "facebook/opt-1.3b")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
KROUTE = 64
KEEPS = [0.50, 0.25, 0.10]


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
    """score:[N,dff] desc-keep top-B per row -> bool keep mask."""
    if B >= score.shape[1]:
        return torch.ones_like(score, dtype=torch.bool)
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def keep_topB_group(score_g, gsz, grp_full, B):
    """score_g:[N,Kc] group scores; keep groups (desc) until cum neuron budget B -> neuron keep mask."""
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, grp_full]


def compute_masked(cfg, a):
    """a:[N,dff] post-ReLU. rep is per-neuron [dff]; select by drop-cost ((a-rep)*vn)^2."""
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
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
    print(f"  MODEL={MODEL}  tokens={ids.numel()}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.decoder.layers; nL = len(layers)
    torch.set_grad_enabled(False)

    CFG = {li: {'active': False} for li in range(nL)}

    def fc2_pre(li):
        def hook(_m, args):
            cfg = CFG[li]
            if not cfg['active']:
                return None
            a = args[0]; sh = a.shape
            masked = compute_masked(cfg, a.reshape(-1, sh[-1]))
            return (masked.reshape(sh),) + args[1:]
        return hook
    for li in range(nL):
        layers[li].fc2.register_forward_pre_hook(fc2_pre(li))

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
    hs = []
    for li in range(nL):
        hs.append(layers[li].fc2.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    act0 = float((torch.cat(capa[0]) > 0).float().mean())
    actM = float((torch.cat(capa[nL // 2]) > 0).float().mean())
    print(f"  OPT: {nL} layers, dff={dff}; ReLU active frac layer0={act0:.3f} mid={actM:.3f}", flush=True)

    STR = {}
    for li in range(nL):
        A = torch.cat(capa[li])                              # [Ncalib, dff] fp16 CPU (kept for condmean)
        a = A.float().to(dev)
        Wup = layers[li].fc1.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        Wd = layers[li].fc2.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0)
        grp_local = kmeans(F.normalize(Wup, dim=1), KROUTE, seed=0)
        Kc = int(grp_local.max().item()) + 1
        gsz = torch.zeros(Kc, device=dev); grp_full = torch.zeros(dff, dtype=torch.long, device=dev)
        for g in range(Kc):
            ix = (grp_local == g).nonzero().flatten()
            gsz[g] = len(ix); grp_full[ix] = g
        STR[li] = dict(A=A, mean=a.mean(0), vn=vn, gsz=gsz, grp_full=grp_full)
        capa[li] = None
        del a, Wup, Wd; gc.collect(); torch.cuda.empty_cache()

    def condmean(li, bf, gran):
        """E[a_k | k dropped] on calibration; dropped set by rep-independent rule keep-top-B(|a|*vn)."""
        s = STR[li]; A = s['A'].float().to(dev); vn = s['vn']; B = int(round(bf * dff))
        score0 = (A.abs() * vn)                              # rep=0 drop-cost magnitude
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
            if reptype == 'mean':
                rep = s['mean']
            elif reptype == 'zero':
                rep = torch.zeros(dff, device=dev)
            else:
                rep = condmean(li, bf, gran)
            CFG[li].update(dict(active=True, B=B, gran=gran, rep=rep, vn=s['vn'],
                                gsz=s['gsz'], grp_full=s['grp_full']))

    def set_active(flag):
        for li in range(nL):
            CFG[li]['active'] = flag

    print(f"\n  OPT(ReLU) ORACLE ppl (dense {dense_ppl:.3f}): unconditional-mean vs CONDITIONAL-mean vs zero\n", flush=True)
    print(f"  {'keep':>5} | {'grp+mean':>9} | {'grp+cond':>9} | {'neu+cond':>9} | {'neu+zero':>9}", flush=True)
    for bf in KEEPS:
        r = {}
        for gran, reptype, key in [('group', 'mean', 'gm'), ('group', 'cond', 'gc'),
                                   ('neuron', 'cond', 'nc'), ('neuron', 'zero', 'nz')]:
            setcfg(bf, gran, reptype); r[key] = ppl(); set_active(False)
        print(f"  {int(bf*100):>4}% | {r['gm']:>9.3f} | {r['gc']:>9.3f} | {r['nc']:>9.3f} | {r['nz']:>9.3f}", flush=True)
    print("\nREAD: if grp+cond/neu+cond << grp+mean (toward dense) => the conditional-mean representative", flush=True)
    print("fixes the collapse; unconditional mean (G-MoE) was the blind spot. neu+zero≈neu+cond confirms", flush=True)
    print("E[a|dropped]≈0 for ReLU. This is a general, activation-agnostic representative improvement.", flush=True)


if __name__ == "__main__":
    main()
