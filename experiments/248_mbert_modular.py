"""Experiment 248 — FAIR mBERT decomposition comparison on XTREME NER + POS (G-MoE Table 1 setup).
Fixes the unfair baseline of exp245: compares grouping strategies at ORACLE selection (no router confound):
  co-activation grouping (faithful MoEfication/G-MoE-style: cluster co-activating neurons) + mean
  keep-pattern grouping (ours, no correction)
  keep-pattern grouping + low-rank correction (ours)
  neuron-oracle (ceiling)
vs dense. Separates the GROUPING gain from the CORRECTION gain. TASK in {ner,pos}. Metric: NER span-F1 / POS acc.
Run: TASK=ner python3 experiments/248_mbert_modular.py   (and TASK=pos)
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = "bert-base-multilingual-cased"
TASK = os.environ.get("TASK", "ner")
LANGS = os.environ.get("LANGS", "en,de,es,fr,nl,ar,zh,hi,ru").split(",")
UDNAME = {"en": "English", "de": "German", "es": "Spanish", "fr": "French", "nl": "Dutch",
          "ar": "Arabic", "zh": "Chinese", "hi": "Hindi", "ru": "Russian"}
SEQ = 128; EPOCHS = int(os.environ.get("EPOCHS", "2"))
K = 128; RCORR = 128; RFEAT = 512
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.75,0.50,0.35").split(",")]
NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
POS_LABELS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
              "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


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
    g = torch.zeros(K, device=dev)
    for j in range(K):
        g[j] = (gl == j).sum()
    return g


def main():
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from datasets import load_dataset, concatenate_datasets
    from seqeval.metrics import f1_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tagfield = "ner_tags" if TASK == "ner" else "pos_tags"

    def load_split(lang, split):
        if TASK == "ner":
            return load_dataset("unimelb-nlp/wikiann", lang, split=split)
        return load_dataset("google/xtreme", f"udpos.{UDNAME[lang]}", split=split)
    tr = concatenate_datasets([load_split(l, "train") for l in LANGS]).shuffle(seed=0)
    te = concatenate_datasets([load_split(l, "test") for l in LANGS])
    # derive label names from the dataset (robust to schema/order)
    feat = tr.features[tagfield]
    names = getattr(getattr(feat, "feature", feat), "names", None)
    LABELS = list(names) if names else (NER_LABELS if TASK == "ner" else POS_LABELS)
    id2lab = {i: l for i, l in enumerate(LABELS)}
    print(f"  TASK={TASK} langs={LANGS} train={len(tr)} test={len(te)}", flush=True)

    def encode(batch):
        toks = batch["tokens"]
        out = tok(toks, is_split_into_words=True, truncation=True, max_length=SEQ, padding="max_length")
        labels = []
        for i, tags in enumerate(batch[tagfield]):
            wids = out.word_ids(i); prev = None; lab = []
            for w in wids:
                if w is None or w == prev:
                    lab.append(-100)
                else:
                    lab.append(int(tags[w]))
                prev = w
            labels.append(lab)
        out["labels"] = labels
        return out
    tr_enc = tr.map(encode, batched=True, remove_columns=tr.column_names)
    te_enc = te.map(encode, batched=True, remove_columns=te.column_names)
    tr_enc.set_format("torch"); te_enc.set_format("torch")
    model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=len(LABELS)).to(dev)

    from torch.utils.data import DataLoader
    dl = DataLoader(tr_enc, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0; nb = 0
        for b in dl:
            o = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev),
                      labels=b["labels"].to(dev))
            opt.zero_grad(); o.loss.backward(); opt.step(); tot += o.loss.item(); nb += 1
        print(f"  epoch {ep} loss {tot/nb:.4f}", flush=True)
    model.eval(); torch.set_grad_enabled(False)

    enc = model.base_model.encoder.layer; nL = len(enc)
    inproj = [enc[li].intermediate.dense for li in range(nL)]
    outproj = [enc[li].output.dense for li in range(nL)]
    CFG = {li: {'active': False} for li in range(nL)}; XC = {li: None for li in range(nL)}

    def gpre(li):
        def h(_m, a):
            XC[li] = a[0].reshape(-1, a[0].shape[-1])
        return h

    def dpre(li):
        def h(_m, args):
            cfg = CFG[li]
            if not cfg['active']:
                return None
            a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1])
            if cfg.get('neuron'):
                m = keep_topB_neuron((af.float() - cfg['abar']).abs() * cfg['vn'], cfg['B']).to(a.dtype)
            else:
                m = oracle_mask(af.float(), cfg['abar'], cfg['vn'], cfg['gsz'], cfg['gf'], cfg['B']).to(a.dtype)
            return ((af * m + cfg['abar'].to(a.dtype) * (1 - m)).reshape(sh),) + args[1:]
        return h

    def dpost(li):
        def h(_m, args, output):
            cfg = CFG[li]
            if not cfg['active'] or not cfg.get('corr'):
                return None
            sh = output.shape
            z = (XC[li].float() - cfg['xbar']) @ cfg['P']
            return (output.reshape(-1, sh[-1]) + (cfg['pred'](z) @ cfg['Br'].T).to(output.dtype)).reshape(sh)
        return h
    for li in range(nL):
        inproj[li].register_forward_pre_hook(gpre(li))
        outproj[li].register_forward_pre_hook(dpre(li))
        outproj[li].register_forward_hook(dpost(li))

    from torch.utils.data import DataLoader as DL
    te_dl = DL(te_enc, batch_size=64); cal_dl = DL(tr_enc.select(range(min(4000, len(tr_enc)))), batch_size=64)

    def metric_eval():
        preds, refs = [], []
        for b in te_dl:
            ii = b["input_ids"].to(dev); am = b["attention_mask"].to(dev); lb = b["labels"]
            pr = model(input_ids=ii, attention_mask=am).logits.argmax(-1).cpu()
            for i in range(ii.shape[0]):
                p, r = [], []
                for j in range(ii.shape[1]):
                    if lb[i, j].item() != -100:
                        p.append(id2lab[pr[i, j].item()]); r.append(id2lab[lb[i, j].item()])
                preds.append(p); refs.append(r)
        if TASK == "ner":
            return f1_score(refs, preds)
        tp = sum(sum(a == b for a, b in zip(p, r)) for p, r in zip(preds, refs))
        tot = sum(len(r) for r in refs)
        return tp / tot

    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}; hs = []
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
        capa[li] = None; capx[li] = None; del a, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  dff={dff} nL={nL}; harvested", flush=True)

    def grouping_coact(li):                                    # faithful MoEfication/G-MoE: cluster co-activating neurons
        s = STR[li]; a = s['a'].float().to(dev)
        pat = F.normalize((a - s['abar']).T.contiguous(), dim=1)   # [m, N] activation pattern per neuron
        gl = balanced_assign(pat, kmeans_centroids(pat, K, seed=0))
        del a, pat; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

    def grouping_keep(li, bf):                                 # ours
        s = STR[li]; a = s['a'].float().to(dev); B = int(round(bf * dff))
        Bind = keep_topB_neuron((a - s['abar']).abs() * s['vn'], B).float()
        w = ((a - s['abar']).abs() * s['vn']).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        del a, Bind; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

    def basis_and_pred(li, bf, gsz, gf):
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev); B = int(round(bf * dff))
        m = oracle_mask(a, abar, s['vn'], gsz, gf, B); drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T; del a, m, drop, Wd; gc.collect(); torch.cuda.empty_cache()
        Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']; pred = mlp_fit(z, E @ Br, dev)
        del E, Ec, Vt, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def setcfg(GRP, B, mode, BR=None, PRED=None):
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=(mode != 'off'), B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                Br=(BR[li] if BR else None), xbar=s['xbar'], P=s['P'], pred=(PRED[li] if PRED else None)))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    unit = "F1" if TASK == "ner" else "acc"
    dense = metric_eval(); print(f"\n  dense {TASK} {unit} = {dense:.4f}\n", flush=True)
    GC = {li: grouping_coact(li) for li in range(nL)}
    for bf in KEEPS:
        B = int(round(bf * dff))
        GP = {li: grouping_keep(li, bf) for li in range(nL)}
        setcfg(GC, B, 'group'); coact = metric_eval(); off()
        setcfg(GP, B, 'group'); kp = metric_eval(); off()
        setcfg(GP, B, 'neuron'); nu = metric_eval(); off()
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = basis_and_pred(li, bf, *GP[li])
        setcfg(GP, B, 'corr', BR, PRED); ours = metric_eval(); off()
        print(f"  FFN{int(bf*100)} [{TASK} {unit}]: dense {dense:.4f} | co-activation {coact:.4f} | "
              f"keep-pattern {kp:.4f} | +corr {ours:.4f} | neuron {nu:.4f}", flush=True)
    print(f"\nREAD: co-activation = faithful MoEfication grouping; keep-pattern = ours (no corr); +corr = ours full.", flush=True)
    print("Separates GROUPING gain (co-act->keep-pattern) from CORRECTION gain (keep-pattern->+corr).", flush=True)


if __name__ == "__main__":
    main()
