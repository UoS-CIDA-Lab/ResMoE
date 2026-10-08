"""Experiment 257 — MoEfication vs ResMoE on a FINE-TUNED T5-large (SST-2 / MNLI / RACE), the
original MoEfication paper's metric (downstream TASK ACCURACY, text-to-text). Same RQ3
component-separation build-up as 254 (group-oracle selection), but eval = task accuracy instead of
denoising perplexity. Reuses 254's grouping/correction/hook machinery verbatim. Rows:
  MoEfication (weight k-means + mean) -> + keep-pattern -> + low-rank correction (ResMoE) -> neuron-oracle
Construction = parameter k-means on wi.weight (K=d_ff/32=128 for large, as in MoEfication); selection
= group-oracle; keep ~20%/25% (MoEfication uses ~20%). HHMODEL should point at a 256 checkpoint
(ckpts/t5-large-<task>); pointing at base t5-large gives the zero-shot reference.
Run: TASK=sst2 HHMODEL=ckpts/t5-large-sst2 conda run -n resmoe python3 experiments/257_t5_glue_corr.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "google-t5/t5-large")
TASK = os.environ.get("TASK", "sst2")
SEQ = int(os.environ.get("SEQ", "512" if TASK == "race" else "128"))
NCAL = int(os.environ.get("NCAL", "4000"))                 # calibration train examples
NEVAL = int(os.environ.get("NEVAL", "0"))                  # 0 = full dev; else cap
CHUNK = int(os.environ.get("CHUNK", "16"))                 # calib micro-batch
MAXROWS = int(os.environ.get("MAXROWS", "16384"))
K = int(os.environ.get("K", "128"))                        # experts = d_ff/32 (MoEfication): small64/base96/large128/3b512
RCORR = 128
RFEAT = 512
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.20,0.25").split(",")]

SST2_LAB = ["negative", "positive"]
MNLI_LAB = ["entailment", "neutral", "contradiction"]


# --- helpers reused verbatim from 254/236 ---
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


def collect_t5_ffns(model):
    assert not getattr(model.config, "is_gated_act", False), \
        "use a NON-gated T5 v1.0 (t5-large), not t5-v1_1/flan (gated-gelu)"
    gproj, downs, tag = [], [], []
    for side, stack in (("enc", model.encoder), ("dec", model.decoder)):
        for bi, blk in enumerate(stack.block):
            ff = blk.layer[-1].DenseReluDense
            assert hasattr(ff, "wi"), "non-gated T5 expected (single wi)"
            gproj.append(ff.wi); downs.append(ff.wo); tag.append(f"{side}{bi}")
    return gproj, downs, tag


def build_task(task, n_eval):
    """Return (calib_srcs, eval_items): calib_srcs=[(src,label_text)], eval_items=[(src,[cands],gold)]."""
    from datasets import load_dataset
    if task == "sst2":
        tr = load_dataset("nyu-mll/glue", "sst2", split="train")
        va = load_dataset("nyu-mll/glue", "sst2", split="validation")
        calib = [(f"sst2 sentence: {r['sentence']}", SST2_LAB[r["label"]]) for r in tr if r["label"] >= 0]
        ev = [(f"sst2 sentence: {r['sentence']}", SST2_LAB, r["label"]) for r in va if r["label"] >= 0]
    elif task == "mnli":
        tr = load_dataset("nyu-mll/glue", "mnli", split="train")
        va = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
        calib = [(f"mnli premise: {r['premise']} hypothesis: {r['hypothesis']}", MNLI_LAB[r["label"]])
                 for r in tr if r["label"] >= 0]
        ev = [(f"mnli premise: {r['premise']} hypothesis: {r['hypothesis']}", MNLI_LAB, r["label"])
              for r in va if r["label"] >= 0]
    elif task == "race":
        tr = load_dataset("ehovy/race", "all", split="train")
        va = load_dataset("ehovy/race", "all", split="test")

        def fmt(r):
            o = " ".join(f"{c}: {t}" for c, t in zip("ABCD", r["options"]))
            return f"question: {r['question']} options: {o} article: {r['article']}"
        calib = [(fmt(r), r["options"]["ABCD".index(r["answer"])]) for r in tr]
        ev = [(fmt(r), r["options"], "ABCD".index(r["answer"])) for r in va]
    else:
        raise ValueError(task)
    if n_eval:
        ev = ev[:n_eval]
    return calib, ev


def main() -> None:
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL, dtype=torch.bfloat16).to(dev).eval()
    model.config.use_cache = False
    gproj, downs, tag = collect_t5_ffns(model); nL = len(gproj)
    n_enc = sum(t.startswith("enc") for t in tag)
    torch.set_grad_enabled(False)

    calib, EVAL = build_task(TASK, NEVAL)
    calib = calib[:NCAL]
    # pre-tokenize calibration (src, gold-label) pairs as teacher-forced forwards
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

    # --- calibration capture over task-train teacher-forced forwards ---
    capa: dict[int, list[torch.Tensor]] = {li: [] for li in range(nL)}
    capx: dict[int, list[torch.Tensor]] = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(downs[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(gproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for b in range(0, len(calib), CHUNK):
        chunk = calib[b:b + CHUNK]
        src = [c[0] for c in chunk]; lab = [c[1] for c in chunk]
        enc = tok(src, return_tensors="pt", padding=True, truncation=True, max_length=SEQ)
        lb = tok(lab, return_tensors="pt", padding=True, truncation=True, max_length=16).input_ids
        lb[lb == tok.pad_token_id] = -100
        model(input_ids=enc.input_ids.to(dev), attention_mask=enc.attention_mask.to(dev), labels=lb.to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]

    def acc_eval():
        correct = 0; tot = 0
        for src, cands, gold in EVAL:
            enc = tok([src], return_tensors="pt", truncation=True, max_length=SEQ)
            C = len(cands)
            lb = tok(list(cands), return_tensors="pt", padding=True, truncation=True, max_length=16).input_ids.to(dev)
            iid = enc.input_ids.to(dev).expand(C, -1)
            am = enc.attention_mask.to(dev).expand(C, -1)
            lab = lb.clone(); lab[lab == tok.pad_token_id] = -100
            logits = model(input_ids=iid, attention_mask=am, labels=lab).logits.float()
            ce = F.cross_entropy(logits.transpose(1, 2), lab.clamp(min=0), reduction='none')  # [C,T]
            mask = (lab != -100).float()
            score = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)                              # length-normalized
            pred = int(score.argmin().item())
            correct += int(pred == gold); tot += 1
        return correct / tot

    STR = {}
    g = torch.Generator().manual_seed(0)
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li])
        if a.shape[0] > MAXROWS:
            idx = torch.randperm(a.shape[0], generator=g)[:MAXROWS]
            a = a[idx]; x = x[idx]
        x = x.float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wg = gproj[li].weight.detach().float().to(dev); Wg = Wg if Wg.shape[0] == dff else Wg.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), Wg=Wg.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li].clear(); capx[li].clear()
        del af, x, Wd, Wg; gc.collect(); torch.cuda.empty_cache()
    print(f"  MODEL={MODEL} TASK={TASK} dff={dff} nL={nL} (enc {n_enc} + dec {nL - n_enc}) "
          f"calib_rows enc={STR[0]['a'].shape[0]} dec={STR[nL-1]['a'].shape[0]} eval={len(EVAL)}", flush=True)

    def grouping_weight(li):                                       # MoEfication: param k-means on wi
        s = STR[li]; Wg = s['Wg'].float().to(dev)
        gl = balanced_assign(Wg, kmeans_centroids(Wg, K, seed=0))
        del Wg; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

    def grouping_keeppattern(li, bf):                             # ResMoE
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

    dense = acc_eval(); print(f"  dense {TASK} acc {dense:.4f}\n", flush=True)
    for bf in KEEPS:
        GW = {li: grouping_weight(li) for li in range(nL)}
        GP = {li: grouping_keeppattern(li, bf) for li in range(nL)}
        setcfg(bf, GW, 'group'); moef = acc_eval(); off()
        setcfg(bf, GP, 'group'); kp = acc_eval(); off()
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = basis_and_pred(li, bf, *GP[li])
        setcfg(bf, GP, 'corr', BR, PRED); res = acc_eval(); off()
        setcfg(bf, GP, 'neuron'); neu = acc_eval(); off()
        print(f"  === keep{int(bf*100)} ({TASK} acc, dense {dense:.4f}) ===", flush=True)
        print(f"    MoEfication (weight k-means + mean)     {moef:.4f}", flush=True)
        print(f"    + keep-pattern grouping                 {kp:.4f}", flush=True)
        print(f"    + low-rank correction (ResMoE)          {res:.4f}", flush=True)
        print(f"    neuron-oracle (ceiling)                 {neu:.4f}\n", flush=True)
    print("READ: task-accuracy build-up over the MoEfication baseline (group-oracle selection),"
          f"\n      on {MODEL} ({TASK}) — the original MoEfication paper's metric.", flush=True)


if __name__ == "__main__":
    main()
