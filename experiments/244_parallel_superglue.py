"""Experiment 244 — PARALLEL-block models on SuperGLUE (G-MoEfication Table 2): Phi-2, Falcon-7B.
Tests our low-rank correction on parallel-block decoders (FFN reads the layer input, parallel to attention),
to REPORT that it works less well there than on sequential blocks (scope boundary, with evidence).
SuperGLUE zero-shot acc (lm_eval) for: dense | group-oracle | +correction | neuron-oracle, keep85.
In-domain (WikiText) calibration. Run: HHMODEL=microsoft/phi-2 python3 experiments/244_parallel_superglue.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "microsoft/phi-2")
N_CALIB = int(os.environ.get("NCALIB", "8192"))
CHUNK = 512
K = 128
RCORR = 128
RFEAT = 512
BF = float(os.environ.get("BF", "0.85"))
TASKS = os.environ.get("TASKS", "boolq,cb,copa,wic,wsc,rte,multirc").split(",")


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


def find_layers(model):
    b = model.model if hasattr(model, "model") else (model.transformer if hasattr(model, "transformer") else model)
    for attr in ("layers", "h", "decoder"):
        if hasattr(b, attr):
            o = getattr(b, attr)
            return o.layers if attr == "decoder" else o
    raise RuntimeError("no layers")


def find_ffn(layer):
    mlp = getattr(layer, "mlp", layer)
    for a, b in [("gate_proj", "down_proj"), ("fc1", "fc2"), ("c_fc", "c_proj"),
                 ("dense_h_to_4h", "dense_4h_to_h")]:
        if hasattr(mlp, a):
            return getattr(mlp, a), getattr(mlp, b)
    raise RuntimeError(f"unknown FFN: {[n for n,_ in mlp.named_children()]}")


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    buf = []
    for t in wt["text"]:
        if t.strip():
            buf.append(t)
        if len(buf) >= 20000:
            break
    ids = tok("\n\n".join(buf), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = find_layers(model); nL = len(layers)
    inproj = []; outproj = []
    for li in range(nL):
        ip, op = find_ffn(layers[li]); inproj.append(ip); outproj.append(op)
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
            o = output[0] if isinstance(output, tuple) else output
            sh = o.shape
            z = (XC[li].float() - cfg['xbar']) @ cfg['P']
            ehat = cfg['pred'](z) @ cfg['Br'].T
            o2 = (o.reshape(-1, sh[-1]) + ehat.to(o.dtype)).reshape(sh)
            return (o2,) + output[1:] if isinstance(output, tuple) else o2
        return hook
    for li in range(nL):
        inproj[li].register_forward_pre_hook(gpre(li))
        outproj[li].register_forward_pre_hook(dpre(li))
        outproj[li].register_forward_hook(dpost(li))

    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(outproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(inproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    B = int(round(BF * dff))
    print(f"  MODEL={MODEL} dff={dff} nL={nL} keep{int(BF*100)} (parallel-block test)", flush=True)

    GS = {}; GF = {}; ABAR = {}; VN = {}; XBAR = {}; P = {}; BR = {}; PRED = {}
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev); x = torch.cat(capx[li]).float().to(dev)
        if a.shape[0] > 16384:
            idx = torch.randperm(a.shape[0])[:16384]; a = a[idx]; x = x[idx]
        Wd = outproj[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
        w = ((a - abar).abs() * vn).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        gsz = torch.zeros(K, device=dev)
        for g in range(K):
            gsz[g] = (gl == g).sum()
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False); Pm = VtX[:RFEAT].T
        m = oracle_mask(a, abar, vn, gsz, gl, B)
        drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T; Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
        pred = mlp_fit((x - xbar) @ Pm, E @ Br, dev)
        GS[li] = gsz; GF[li] = gl; ABAR[li] = abar; VN[li] = vn; XBAR[li] = xbar; P[li] = Pm; BR[li] = Br; PRED[li] = pred
        capa[li] = None; capx[li] = None
        del a, x, Wd, Bind, m, drop, E, Ec, Vt; gc.collect(); torch.cuda.empty_cache()
    print("  built grouping+correction.\n", flush=True)

    def gen_eval(tag):
        model.config.use_cache = True
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=8)
        r = lm_eval.simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, verbosity="ERROR")
        accs = {}
        for t in TASKS:
            res = r['results'].get(t, {})
            accs[t] = res.get('acc,none', res.get('acc_norm,none'))
        vals = [v for v in accs.values() if v is not None]
        mean = sum(vals) / max(len(vals), 1)
        line = " ".join(f"{t}={accs[t]:.3f}" for t in TASKS if accs[t] is not None)
        print(f"  [{tag}] {line} | MEAN={mean:.3f}", flush=True)
        model.config.use_cache = False
        del lm, r; gc.collect(); torch.cuda.empty_cache()
        return mean

    def setcfg(mode):
        for li in range(nL):
            CFG[li].update(dict(active=(mode != 'dense'), B=B, vn=VN[li], abar=ABAR[li], gsz=GS[li], gf=GF[li],
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                Br=BR[li], xbar=XBAR[li], P=P[li], pred=PRED[li]))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    setcfg('dense'); d = gen_eval("dense"); off()
    setcfg('group'); g = gen_eval(f"keep{int(BF*100)} group-oracle"); off()
    setcfg('corr'); c = gen_eval(f"keep{int(BF*100)} +correction"); off()
    setcfg('neuron'); nu = gen_eval(f"keep{int(BF*100)} neuron-oracle"); off()
    print(f"\n  ==> {MODEL} keep{int(BF*100)}: dense {d:.3f} | group {g:.3f} | +corr {c:.3f} | neuron {nu:.3f}", flush=True)
    print("READ: parallel-block. If +corr recovers LESS of the floor than on sequential blocks =>", flush=True)
    print("evidence that the method is weaker on parallel-block FFNs (scope boundary).", flush=True)


if __name__ == "__main__":
    main()
