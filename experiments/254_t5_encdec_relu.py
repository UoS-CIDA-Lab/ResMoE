"""Experiment 254 — low-rank output correction on an ENCODER-DECODER (T5, ReLU): the architecture
absent from the rest of the study (all others are decoder-only or BERT encoder). T5 v1.0 is the
ORIGINAL MoEfication's home turf (ReLU, seq2seq). Same RQ3 "component-separation" protocol as 236
(tab:baseline), at group-ORACLE selection (no router confound), perplexity, NO fine-tuning. Rows:
  MoEfication/G-MoE grouping  : balanced k-means on wi INPUT-WEIGHT rows (parameter clustering) + mean rep
  + keep-pattern grouping     : our co-keep-pattern clustering + mean rep (isolates the grouping gain)
  + low-rank correction (Ours): keep-pattern + predicted rank-r correction (isolates the correction gain)
  neuron-oracle               : per-neuron ceiling
Every FFN (encoder.block[i].layer[1] AND decoder.block[i].layer[2]) is treated uniformly. Since T5 v1.0
has no causal LM, "perplexity" = DENOISING (span-corruption) perplexity: corrupt source spans, teacher-
force the decoder on the sentinel-delimited target, exp(mean CE). The SAME corrupted batches are reused
across every config for a fair comparison. Use a NON-gated v1.0 checkpoint (t5-base/large/3b), NOT t5-v1_1
/flan (gated-gelu, two wi). bf16 (T5 is fp16-unstable). NCALIB/NEVAL are WINDOW counts here.
Run: HHMODEL=t5-base conda run -n resmoe python3 experiments/254_t5_encdec_relu.py
"""
from __future__ import annotations
import sys, pathlib, gc, os, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "t5-base")
SEQ = int(os.environ.get("SEQ", "512"))
CHUNK = int(os.environ.get("CHUNK", "8"))          # batch of sequences per forward
NCAL_WIN = int(os.environ.get("NCALIB", "48"))     # calibration windows
NEVAL_WIN = int(os.environ.get("NEVAL", "48"))     # eval windows
MAXROWS = int(os.environ.get("MAXROWS", "16384"))  # per-layer calib-row cap (memory guard, cf. 248)
DENSITY = float(os.environ.get("NOISE", "0.15"))   # span-corruption noise density
MEANSPAN = int(os.environ.get("MEANSPAN", "3"))    # mean noise span length
K = 128
RCORR = 128
RFEAT = 512
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.50,0.25").split(",")]


# ---------------------------------------------------------------------------
# Reused verbatim from 236_baseline_compare.py (activation-agnostic helpers).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# T5-specific: FFN enumeration + span-corruption (denoising) data pipeline.
# ---------------------------------------------------------------------------
def collect_t5_ffns(model):
    """Flat parallel lists over EVERY FFN (encoder + decoder). gproj/downs mirror the
    236 roles: gproj=wi (FFN input proj, captures x), downs=wo (output proj, input=a)."""
    assert not getattr(model.config, "is_gated_act", False), \
        "use a NON-gated T5 v1.0 (t5-base/large/3b), not t5-v1_1/flan (gated-gelu)"
    gproj, downs, tag = [], [], []
    for side, stack in (("enc", model.encoder), ("dec", model.decoder)):
        for bi, blk in enumerate(stack.block):
            ff = blk.layer[-1].DenseReluDense            # FF is always layer[-1]
            assert hasattr(ff, "wi"), "non-gated T5 expected (single wi)"
            gproj.append(ff.wi); downs.append(ff.wo); tag.append(f"{side}{bi}")
    return gproj, downs, tag


def _random_segmentation(num_items, num_segments, rng):
    """Partition num_items into num_segments non-empty segments; return segment lengths."""
    if num_segments >= num_items:
        return np.ones(num_items, dtype=np.int64) if num_items == num_segments \
            else np.bincount(np.arange(num_items) % num_segments, minlength=num_segments)
    breaks = np.arange(num_items - 1) < (num_segments - 1)
    rng.shuffle(breaks)
    first_in_seg = np.concatenate([[0], breaks.astype(np.int64)])
    seg_id = np.cumsum(first_in_seg)
    return np.bincount(seg_id, minlength=num_segments)


def random_spans_noise_mask(length, density, mean_span, rng):
    """Standard T5 span-corruption mask. True where the token is noise (to be masked)."""
    num_noise = int(round(length * density))
    num_noise = min(max(num_noise, 1), length - 1)
    num_spans = max(int(round(num_noise / mean_span)), 1)
    num_nonnoise = length - num_noise
    noise_len = _random_segmentation(num_noise, num_spans, rng)
    nonnoise_len = _random_segmentation(num_nonnoise, num_spans, rng)
    interleaved = np.stack([nonnoise_len, noise_len], axis=1).reshape(-1)  # nonnoise first
    span_starts = np.cumsum(interleaved)[:-1]
    indicator = np.zeros(length, dtype=np.int64)
    indicator[span_starts] = 1
    return (np.cumsum(indicator) % 2) == 1


def _corrupt_one(toks, is_noise, sentinels, eos):
    """Replace each maximal noise span with one sentinel (inputs); emit sentinel+span tokens (targets)."""
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
    """Build a fixed list of teacher-forcing batches once; reuse across ALL configs."""
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

    # --- corpus -> SEQ-token windows -> disjoint calib / eval corrupted batches ---
    ds = load_dataset("wikitext", os.environ.get("WIKI", "wikitext-103-raw-v1"), split="test")
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

    # --- calibration capture (enc+dec FFNs fire in one teacher-forced forward) ---
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
        if a.shape[0] > MAXROWS:                                   # joint subsample (a,x share rows)
            idx = torch.randperm(a.shape[0], generator=g)[:MAXROWS]
            a = a[idx]; x = x[idx]
        x = x.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wg = gproj[li].weight.detach().float().to(dev)             # [m,d] wi input-weight rows
        Wg = Wg if Wg.shape[0] == dff else Wg.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), Wg=Wg.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del af, x, Wd, Wg; gc.collect(); torch.cuda.empty_cache()
    print(f"  MODEL={MODEL} dff={dff} nL={nL} (enc {n_enc} + dec {nL - n_enc}) "
          f"calib_rows enc={STR[0]['a'].shape[0]} dec={STR[nL-1]['a'].shape[0]}", flush=True)

    def grouping_weight(li):                                       # G-MoE/MoEfication: param clustering
        s = STR[li]; Wg = s['Wg'].float().to(dev)
        gl = balanced_assign(Wg, kmeans_centroids(Wg, K, seed=0))
        del Wg; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

    def grouping_keeppattern(li, bf):                             # ours
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

    dense = ce_eval(); print(f"  dense denoising-ppl {dense:.3f}\n", flush=True)
    for bf in KEEPS:
        GW = {li: grouping_weight(li) for li in range(nL)}
        GP = {li: grouping_keeppattern(li, bf) for li in range(nL)}
        setcfg(bf, GW, 'group'); gmoe = ce_eval(); off()
        setcfg(bf, GP, 'group'); ours_grp = ce_eval(); off()
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = basis_and_pred(li, bf, *GP[li])
        setcfg(bf, GP, 'corr', BR, PRED); ours_full = ce_eval(); off()
        setcfg(bf, GP, 'neuron'); neu = ce_eval(); off()
        print(f"  === keep{int(bf*100)} (dense {dense:.3f}) ===", flush=True)
        print(f"    G-MoE/MoEfication grouping (weight k-means + mean)  {gmoe:.3f}", flush=True)
        print(f"    + keep-pattern grouping (ours grouping)             {ours_grp:.3f}", flush=True)
        print(f"    + low-rank correction (OURS full = ResMoE)          {ours_full:.3f}", flush=True)
        print(f"    neuron-oracle (ceiling)                             {neu:.3f}\n", flush=True)
    print("READ: each row adds one of our components over the G-MoE baseline (all at oracle selection),"
          "\n      on an encoder-decoder (T5/ReLU) — the architecture absent from the rest of the study.", flush=True)


if __name__ == "__main__":
    main()
