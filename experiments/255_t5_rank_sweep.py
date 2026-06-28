"""Experiment 255 — correction-RANK sweep on an ENCODER-DECODER (T5, ReLU), the enc-dec analog of
the paper's rank figure (fig:rank, originally mBERT keep25). ONE model, keep25, group-ORACLE
selection (keep-pattern grouping), denoising (span-corruption) perplexity, NO fine-tuning. Reports
Dense / Floor (group-oracle, no correction) / Ceiling (neuron-oracle) and ResMoE at each correction
rank r in RANKS, to show the dropped-output error is low-rank (curve saturates by r=128 and crosses
the floor). Shares 254's scaffolding (FFN enumeration, span-corruption pipeline, calibration, eval).
Use a NON-gated v1.0 checkpoint (t5-base/large/3b), bf16. NCALIB/NEVAL are WINDOW counts.
Run: HHMODEL=t5-large conda run -n resmoe python3 experiments/255_t5_rank_sweep.py
"""
from __future__ import annotations
import sys, pathlib, gc, os, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "t5-large")
SEQ = int(os.environ.get("SEQ", "512"))
CHUNK = int(os.environ.get("CHUNK", "8"))
NCAL_WIN = int(os.environ.get("NCALIB", "128"))
NEVAL_WIN = int(os.environ.get("NEVAL", "64"))
MAXROWS = int(os.environ.get("MAXROWS", "16384"))
DENSITY = float(os.environ.get("NOISE", "0.15"))
MEANSPAN = int(os.environ.get("MEANSPAN", "3"))
K = 128
RFEAT = 512
RANK_KEEP = float(os.environ.get("RANK_KEEP", "0.25"))                     # fig:rank uses keep25
RANKS = [int(x) for x in os.environ.get("RANKS", "16,32,64,128").split(",") if x.strip()]


# --- helpers reused verbatim from 254/236 ---
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


def collect_t5_ffns(model):
    assert not getattr(model.config, "is_gated_act", False), \
        "use a NON-gated T5 v1.0 (t5-base/large/3b), not t5-v1_1/flan (gated-gelu)"
    gproj, downs, tag = [], [], []
    for side, stack in (("enc", model.encoder), ("dec", model.decoder)):
        for bi, blk in enumerate(stack.block):
            ff = blk.layer[-1].DenseReluDense
            assert hasattr(ff, "wi"), "non-gated T5 expected (single wi)"
            gproj.append(ff.wi); downs.append(ff.wo); tag.append(f"{side}{bi}")
    return gproj, downs, tag


def _random_segmentation(num_items, num_segments, rng):
    if num_segments >= num_items:
        return np.ones(num_items, dtype=np.int64) if num_items == num_segments \
            else np.bincount(np.arange(num_items) % num_segments, minlength=num_segments)
    breaks = np.arange(num_items - 1) < (num_segments - 1)
    rng.shuffle(breaks)
    first_in_seg = np.concatenate([[0], breaks.astype(np.int64)])
    seg_id = np.cumsum(first_in_seg)
    return np.bincount(seg_id, minlength=num_segments)


def random_spans_noise_mask(length, density, mean_span, rng):
    num_noise = int(round(length * density))
    num_noise = min(max(num_noise, 1), length - 1)
    num_spans = max(int(round(num_noise / mean_span)), 1)
    num_nonnoise = length - num_noise
    noise_len = _random_segmentation(num_noise, num_spans, rng)
    nonnoise_len = _random_segmentation(num_nonnoise, num_spans, rng)
    interleaved = np.stack([nonnoise_len, noise_len], axis=1).reshape(-1)
    span_starts = np.cumsum(interleaved)[:-1]
    indicator = np.zeros(length, dtype=np.int64)
    indicator[span_starts] = 1
    return (np.cumsum(indicator) % 2) == 1


def _corrupt_one(toks, is_noise, sentinels, eos):
    inp, tgt = [], []
    si = 0; prev_noise = False
    for t, noise in zip(toks, is_noise):
        if noise:
            if not prev_noise:
                inp.append(sentinels[si]); tgt.append(sentinels[si]); si += 1
            tgt.append(int(t)); prev_noise = True
        else:
            inp.append(int(t)); prev_noise = False
    inp.append(eos); tgt.append(eos)
    return inp, tgt


def build_corrupted_batches(tok, windows, batch_size, density, mean_span, seed):
    sentinels = [tok.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)]
    eos = tok.eos_token_id; pad = tok.pad_token_id
    rng = np.random.default_rng(seed)
    exs = []
    for w in windows:
        is_noise = random_spans_noise_mask(len(w), density, mean_span, rng)
        exs.append(_corrupt_one(w, is_noise, sentinels, eos))
    batches = []
    for b0 in range(0, len(exs), batch_size):
        chunk = exs[b0:b0 + batch_size]
        mi = max(len(e[0]) for e in chunk); mt = max(len(e[1]) for e in chunk)
        input_ids = torch.full((len(chunk), mi), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), mi), dtype=torch.long)
        labels = torch.full((len(chunk), mt), -100, dtype=torch.long)
        for j, (inp, tgt) in enumerate(chunk):
            input_ids[j, :len(inp)] = torch.tensor(inp); attn[j, :len(inp)] = 1
            labels[j, :len(tgt)] = torch.tensor(tgt)
        batches.append(dict(input_ids=input_ids, attention_mask=attn, labels=labels))
    return batches


def main():
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    model.config.use_cache = False
    gproj, downs, tag = collect_t5_ffns(model); nL = len(gproj)
    n_enc = sum(t.startswith("enc") for t in tag)
    torch.set_grad_enabled(False)

    ds = load_dataset(os.environ.get("WIKI_REPO", "Salesforce/wikitext"),
                      os.environ.get("WIKI", "wikitext-103-raw-v1"), split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    wid = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    need = NCAL_WIN + NEVAL_WIN
    nwin = min(wid.numel() // SEQ, need)
    assert nwin >= need, f"corpus too small: {nwin} < {need} windows"
    windows = wid[:nwin * SEQ].view(nwin, SEQ).tolist()
    CALIB_BATCHES = build_corrupted_batches(tok, windows[:NCAL_WIN], CHUNK, DENSITY, MEANSPAN, seed=0)
    EVAL_BATCHES = build_corrupted_batches(tok, windows[NCAL_WIN:NCAL_WIN + NEVAL_WIN], CHUNK, DENSITY, MEANSPAN, seed=1)

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
    for b in CALIB_BATCHES:
        model(input_ids=b['input_ids'].to(dev), attention_mask=b['attention_mask'].to(dev),
              labels=b['labels'].to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]

    def ce_eval():
        tot = 0.0; n = 0
        for b in EVAL_BATCHES:
            out = model(input_ids=b['input_ids'].to(dev), attention_mask=b['attention_mask'].to(dev),
                        labels=b['labels'].to(dev))
            lbl = b['labels'].to(dev); mask = lbl != -100
            lo = out.logits.float()[mask]
            tot += F.cross_entropy(lo, lbl[mask], reduction='sum').item(); n += int(mask.sum())
        return math.exp(tot / n)

    STR = {}
    g = torch.Generator().manual_seed(0)
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li])
        if a.shape[0] > MAXROWS:
            idx = torch.randperm(a.shape[0], generator=g)[:MAXROWS]
            a = a[idx]; x = x[idx]
        x = x.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del af, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  MODEL={MODEL} dff={dff} nL={nL} (enc {n_enc} + dec {nL - n_enc}) "
          f"keep{int(RANK_KEEP*100)} ranks={RANKS}", flush=True)

    def grouping_keeppattern(li, bf):
        s = STR[li]; a = s['a'].float().to(dev); vn = s['vn']; abar = s['abar']; B = int(round(bf * dff))
        Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
        w = ((a - abar).abs() * vn).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        del a, Bind; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

    def setcfg(bf, GRP, mode, BR=None, PRED=None):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=(mode != 'off'), B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                Br=(BR[li] if BR else None), xbar=s['xbar'], P=s['P'],
                                pred=(PRED[li] if PRED else None)))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    # --- rank sweep at a single keep (fig:rank analog) ---
    bf = RANK_KEEP
    dense = ce_eval()
    GP = {li: grouping_keeppattern(li, bf) for li in range(nL)}
    setcfg(bf, GP, 'group'); floor = ce_eval(); off()
    setcfg(bf, GP, 'neuron'); ceil = ce_eval(); off()

    # per-layer E + SVD computed ONCE; MLP refit per rank (output dim = rank)
    BR = {r: {} for r in RANKS}; PRED = {r: {} for r in RANKS}
    rmax = max(RANKS)
    for li in range(nL):
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        m = oracle_mask(a, abar, s['vn'], GP[li][0], GP[li][1], B)
        drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T
        del a, m, drop, Wd; gc.collect(); torch.cuda.empty_cache()
        Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        for r in RANKS:
            Br = Vt[:r].T.to(dev); c = E @ Br
            BR[r][li] = Br; PRED[r][li] = mlp_fit(z, c, dev)
            del Br, c
        del E, Ec, Vt, z; gc.collect(); torch.cuda.empty_cache()

    resmoe = {}
    for r in RANKS:
        setcfg(bf, GP, 'corr', BR[r], PRED[r]); resmoe[r] = ce_eval(); off()

    print(f"\n  === rank sweep (keep{int(bf*100)}, {MODEL}) — denoising perplexity ===", flush=True)
    print(f"    Dense                         {dense:.3f}", flush=True)
    print(f"    Floor (group-oracle, no corr) {floor:.3f}", flush=True)
    for r in RANKS:
        print(f"    ResMoE  rank {r:4d}            {resmoe[r]:.3f}", flush=True)
    print(f"    Ceiling (neuron-oracle)       {ceil:.3f}", flush=True)
    print("READ: ResMoE perplexity vs correction rank at keep25 (enc-dec analog of fig:rank);"
          "\n      expect it to drop from the Floor toward the Ceiling and saturate by r=128.", flush=True)


if __name__ == "__main__":
    main()
