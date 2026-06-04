"""Experiment 245 — reproduce G-MoEfication Table 1 (mBERT XTREME NER) with OUR method.
Fine-tune mBERT on wikiann NER (representative multilingual subset of XTREME), then MoEfy the FFN at
FFN 35/50/75 and compare: G-MoE grouping (weight k-means + mean) vs OURS (keep-pattern + low-rank correction),
vs dense and neuron-oracle. Metric: seqeval span-F1. In-domain calibration (NER train text).
Run: python3 experiments/245_mbert_ner.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = "bert-base-multilingual-cased"
LANGS = os.environ.get("LANGS", "en,de,es,fr,nl,ar,zh,hi,ru").split(",")
SEQ = 128
EPOCHS = int(os.environ.get("EPOCHS", "2"))
K = 128
RCORR = 128
RFEAT = 512
KEEPS = [0.75, 0.50, 0.35]                                   # G-MoE FFN 75/50/35
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id2lab = {i: l for i, l in enumerate(LABELS)}


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
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from datasets import load_dataset, concatenate_datasets
    from seqeval.metrics import f1_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)

    def load_lang(lang, split):
        return load_dataset("wikiann", lang, split=split, trust_remote_code=True)
    tr = concatenate_datasets([load_lang(l, "train") for l in LANGS]).shuffle(seed=0)
    te = concatenate_datasets([load_lang(l, "test") for l in LANGS])
    print(f"  langs={LANGS} train={len(tr)} test={len(te)}", flush=True)

    def encode(batch):
        out = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=SEQ, padding="max_length")
        labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            wids = out.word_ids(i); prev = None; lab = []
            for w in wids:
                if w is None:
                    lab.append(-100)
                elif w != prev:
                    lab.append(tags[w])
                else:
                    lab.append(-100)
                prev = w
            labels.append(lab)
        out["labels"] = labels
        return out
    tr_enc = tr.map(encode, batched=True, remove_columns=tr.column_names)
    te_enc = te.map(encode, batched=True, remove_columns=te.column_names)
    tr_enc.set_format("torch"); te_enc.set_format("torch")

    model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=len(LABELS)).to(dev)

    # ---- fine-tune ----
    from torch.utils.data import DataLoader
    dl = DataLoader(tr_enc, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0; nb = 0
        for b in dl:
            ii = b["input_ids"].to(dev); am = b["attention_mask"].to(dev); lb = b["labels"].to(dev)
            opt.zero_grad(); out = model(input_ids=ii, attention_mask=am, labels=lb)
            out.loss.backward(); opt.step(); tot += out.loss.item(); nb += 1
        print(f"  epoch {ep} loss {tot/nb:.4f}", flush=True)
    model.eval()
    torch.set_grad_enabled(False)

    enc = model.base_model.encoder.layer; nL = len(enc)
    inproj = [enc[li].intermediate.dense for li in range(nL)]
    outproj = [enc[li].output.dense for li in range(nL)]
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
                m = keep_topB_neuron((af.float() - cfg['abar']).abs() * cfg['vn'], cfg['B']).to(a.dtype)
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

    from torch.utils.data import DataLoader as DL
    te_dl = DL(te_enc, batch_size=64)
    tr_small = tr_enc.select(range(min(4000, len(tr_enc))))
    cal_dl = DL(tr_small, batch_size=64)

    def f1_eval():
        preds, refs = [], []
        for b in te_dl:
            ii = b["input_ids"].to(dev); am = b["attention_mask"].to(dev); lb = b["labels"]
            logits = model(input_ids=ii, attention_mask=am).logits.argmax(-1).cpu()
            for i in range(ii.shape[0]):
                p, r = [], []
                for j in range(ii.shape[1]):
                    if lb[i, j].item() != -100:
                        p.append(id2lab[logits[i, j].item()]); r.append(id2lab[lb[i, j].item()])
                preds.append(p); refs.append(r)
        return f1_score(refs, preds)

    # ---- harvest calib activations (in-domain NER train text) ----
    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(outproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(inproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for b in cal_dl:
        model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"  dff={dff} nL={nL}; harvested calib", flush=True)

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li])
        if a.shape[0] > 16384:
            idx = torch.randperm(a.shape[0])[:16384]; a = a[idx]; x = x[idx]
        a = a.float().to(dev); x = x.float().to(dev)
        Wd = outproj[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wg = inproj[li].weight.detach().float().to(dev); Wg = Wg if Wg.shape[0] == dff else Wg.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a.half().cpu(), x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), Wg=Wg.cpu(),
                       xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del a, x, Wd, Wg; gc.collect(); torch.cuda.empty_cache()

    def grouping_weight(li):
        s = STR[li]; Wg = s['Wg'].float().to(dev)
        gl = balanced_assign(Wg, kmeans_centroids(Wg, K, seed=0))
        del Wg; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

    def grouping_keep(li, bf):
        s = STR[li]; a = s['a'].float().to(dev); vn = s['vn']; abar = s['abar']; B = int(round(bf * dff))
        Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
        w = ((a - abar).abs() * vn).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        del a, Bind; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

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

    def setcfg(GRP, B, mode, BR=None, PRED=None):
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=(mode != 'off'), B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                Br=(BR[li] if BR else None), xbar=s['xbar'], P=s['P'],
                                pred=(PRED[li] if PRED else None)))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    dense = f1_eval(); print(f"\n  dense NER F1 = {dense:.4f}\n", flush=True)
    GW = {li: grouping_weight(li) for li in range(nL)}
    for bf in KEEPS:
        B = int(round(bf * dff))
        GP = {li: grouping_keep(li, bf) for li in range(nL)}
        setcfg(GW, B, 'group'); gmoe = f1_eval(); off()
        setcfg(GP, B, 'neuron'); nu = f1_eval(); off()
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = basis_and_pred(li, bf, *GP[li])
        setcfg(GP, B, 'corr', BR, PRED); ours = f1_eval(); off()
        print(f"  FFN{int(bf*100)}: dense {dense:.4f} | G-MoE-grouping {gmoe:.4f} | OURS(keep+corr) {ours:.4f} | neuron {nu:.4f}", flush=True)
    print("\nREAD: mBERT NER (XTREME subset) reproduction. OURS(keep-pattern+correction) vs G-MoE grouping at FFN35/50/75.", flush=True)


if __name__ == "__main__":
    main()
