"""Experiment 237 — BENCHMARK baseline comparison on HumanEval (Qwen2.5-Coder-1.5B, keep50).
Adds the TRUE G-MoEfication/MoEfication grouping (balanced k-means on input-weight rows + mean rep)
as a baseline row, vs our keep-pattern grouping + low-rank correction, all at group-oracle selection.
Modes (pass@1): dense | G-MoE grouping (weight k-means, no corr) | ours keep-pattern + corr | neuron-oracle.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/237_humaneval_baseline.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
CHUNK = 512
K = 128
RCORR = 128
BF = 0.50


def kmeans_centroids(X, k, iters=20, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return c


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


def mlp_fit(X, Y, dev, steps=3000, hidden=512, lr=3e-3, bs=2048):
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


def gsizes(gl, dev):
    gsz = torch.zeros(K, device=dev)
    for g in range(K):
        gsz[g] = (gl == g).sum()
    return gsz


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
            ehat = cfg['pred']((x - cfg['xbar']) @ cfg['P']) @ cfg['Br'].T
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

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
    dff = capa[0][0].shape[1]
    B = int(round(BF * dff))
    print(f"  MODEL={MODEL} dff={dff}; building groupings+correction @keep{int(BF*100)}", flush=True)

    GW = {}; GP = {}; ABAR = {}; VN = {}; XBAR = {}; P = {}; BR = {}; PRED = {}
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wg = gproj[li].weight.detach().float().to(dev); Wg = Wg if Wg.shape[0] == dff else Wg.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        # G-MoE grouping: param k-means on gate input-weight rows
        glw = balanced_assign(Wg, kmeans_centroids(Wg, K, seed=0))
        # ours: keep-pattern grouping
        Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
        w = ((a - abar).abs() * vn).mean(0)
        glp = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False); Pm = VtX[:512].T
        # correction built on OUR grouping
        m = oracle_mask(a, abar, vn, gsizes(glp, dev), glp, B)
        drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T; _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
        pred = mlp_fit((x - xbar) @ Pm, E @ Br, dev)
        GW[li] = (gsizes(glw, dev), glw); GP[li] = (gsizes(glp, dev), glp)
        ABAR[li] = abar; VN[li] = vn; XBAR[li] = xbar; P[li] = Pm; BR[li] = Br; PRED[li] = pred
        capa[li] = None; capx[li] = None
        del a, x, Wd, Wg, Bind, m, drop, E, Vt; gc.collect(); torch.cuda.empty_cache()
    print("  built. running HumanEval...\n", flush=True)

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    def gen_eval(tag):
        model.config.use_cache = True
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1)
        r = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"], num_fewshot=0,
                                    confirm_run_unsafe_code=True, verbosity="ERROR")
        res = r['results']['humaneval']
        p1 = next((v for k, v in res.items() if k.startswith('pass@1')), None)
        print(f"  [{tag}] HumanEval pass@1 = {p1}", flush=True)
        model.config.use_cache = False
        del lm, r; gc.collect(); torch.cuda.empty_cache()
        return p1

    def setcfg(GRP, neuron=False, corr=False):
        for li in range(nL):
            gsz, gf = GRP[li] if GRP else (None, None)
            CFG[li].update(dict(active=(GRP is not None), B=B, abar=ABAR[li], vn=VN[li], gsz=gsz, gf=gf,
                                neuron=neuron, corr=corr, xbar=XBAR[li], P=P[li], Br=BR[li], pred=PRED[li]))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    setcfg(None); gen_eval("dense"); off()
    setcfg(GW); gen_eval("G-MoE grouping (weight k-means, no corr)"); off()
    setcfg(GP, corr=True); gen_eval("OURS keep-pattern + correction"); off()
    setcfg(GP, neuron=True); gen_eval("neuron-oracle (ceiling)"); off()
    print("\nREAD: ours (keep-pattern+corr) vs the G-MoE grouping baseline on HumanEval pass@1, keep50.", flush=True)


if __name__ == "__main__":
    main()
