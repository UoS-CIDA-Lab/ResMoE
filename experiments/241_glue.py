"""Experiment 241 — GLUE encoder downstream (the original MoEfication/G-MoEfication encoder benchmark).
Fine-tuned BERT per task (textattack/bert-base-uncased-{SST-2,MRPC,QNLI}) + our FFN low-rank correction,
in-domain calibration (task train text). Dev accuracy for:
dense | group-oracle (no corr) | group-oracle + predicted rank-128 correction | neuron-oracle.
Run: TASK=sst2 python3 experiments/241_glue.py   (TASK in {sst2,mrpc,qnli})
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

TASK = os.environ.get("TASK", "sst2")
CKPT = {"sst2": "textattack/bert-base-uncased-SST-2",
        "mrpc": "textattack/bert-base-uncased-MRPC",
        "qnli": "textattack/bert-base-uncased-QNLI"}[TASK]
FIELDS = {"sst2": ("sentence", None), "mrpc": ("sentence1", "sentence2"),
          "qnli": ("question", "sentence")}[TASK]
SEQ = 128
N_CALIB = int(os.environ.get("NCALIB", "4000"))
N_EVAL = int(os.environ.get("NEVAL", "100000"))
K = 128
RCORR = 128
RFEAT = 512
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.50,0.25").split(",")]


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


def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(CKPT)
    ds = load_dataset("nyu-mll/glue", TASK)
    f1, f2 = FIELDS
    tr, va = ds["train"], ds["validation"]
    n_tr = min(N_CALIB, len(tr)); n_va = min(N_EVAL, len(va))
    calib = [(tr[i][f1], tr[i][f2] if f2 else None) for i in range(n_tr)]
    evalset = [(va[i][f1], va[i][f2] if f2 else None) for i in range(n_va)]
    eval_lab = torch.tensor([va[i]["label"] for i in range(n_va)])
    model = AutoModelForSequenceClassification.from_pretrained(CKPT, dtype=torch.float16).to(dev).eval()
    enc = model.base_model.encoder.layer; nL = len(enc)
    inproj = [enc[li].intermediate.dense for li in range(nL)]
    outproj = [enc[li].output.dense for li in range(nL)]
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
            sh = output.shape
            z = (XC[li].float() - cfg['xbar']) @ cfg['P']
            ehat = cfg['pred'](z) @ cfg['Br'].T
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        inproj[li].register_forward_pre_hook(gpre(li))
        outproj[li].register_forward_pre_hook(dpre(li))
        outproj[li].register_forward_hook(dpost(li))

    def batches(pairs, bs=64):
        for b in range(0, len(pairs), bs):
            chunk = pairs[b:b + bs]
            a = [c[0] for c in chunk]; bb = [c[1] for c in chunk]
            if bb[0] is None:
                enc_in = tok(a, return_tensors="pt", truncation=True, max_length=SEQ, padding=True)
            else:
                enc_in = tok(a, bb, return_tensors="pt", truncation=True, max_length=SEQ, padding=True)
            yield {k: v.to(dev) for k, v in enc_in.items()}

    # ---- harvest (in-domain: task train text) ----
    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(outproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(inproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for bt in batches(calib):
        model(**bt)
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]

    def acc_eval():
        preds = []
        for bt in batches(evalset):
            preds.append(model(**bt).logits.argmax(-1).cpu())
        pred = torch.cat(preds)
        return (pred == eval_lab[:len(pred)]).float().mean().item()

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li])
        if a.shape[0] > 16384:
            idx = torch.randperm(a.shape[0])[:16384]; a = a[idx]; x = x[idx]
        a = a.float().to(dev); x = x.float().to(dev)
        Wd = outproj[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a.half().cpu(), x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del a, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  TASK={TASK} ckpt={CKPT} dff={dff} nL={nL} (n_eval={n_va})", flush=True)

    def grouping(li, bf):
        s = STR[li]; a = s['a'].float().to(dev); vn = s['vn']; abar = s['abar']; B = int(round(bf * dff))
        Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
        w = ((a - abar).abs() * vn).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        gsz = torch.zeros(K, device=dev)
        for g in range(K):
            gsz[g] = (gl == g).sum()
        del a, Bind; gc.collect(); torch.cuda.empty_cache()
        return gsz, gl

    def basis_and_pred(li, bf, gsz, gf):
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        m = oracle_mask(a, abar, s['vn'], gsz, gf, B)
        drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T
        del a, m, drop, Wd; gc.collect(); torch.cuda.empty_cache()
        Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
        c = E @ Br
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        pred = mlp_fit(z, c, dev)
        del E, Ec, Vt, c, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def setcfg(bf, GRP, mode, Br=None, pred=None):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=(mode != 'off'), B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                Br=(Br[li] if Br else None), xbar=s['xbar'], P=s['P'],
                                pred=(pred[li] if pred else None)))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    dense = acc_eval(); print(f"  dense acc {dense:.4f}\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        setcfg(bf, GRP, 'group'); floor = acc_eval(); off()
        setcfg(bf, GRP, 'neuron'); neu = acc_eval(); off()
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = basis_and_pred(li, bf, *GRP[li])
        setcfg(bf, GRP, 'corr', BR, PRED); corr = acc_eval(); off()
        print(f"  [{TASK}] keep{int(bf*100)}: dense {dense:.4f} | neu {neu:.4f} | group {floor:.4f} | +corr {corr:.4f}", flush=True)
    print(f"\nREAD: {TASK} dev accuracy; +corr > group => correction helps the GLUE task (in-domain calib).", flush=True)


if __name__ == "__main__":
    main()
