"""Experiment 256 — MultiPL-E Java pass@k for ALL rows of the G-MoEfication benchmark table under ONE
sampling protocol (Java-fine-tuned SantaCoder, exp251 checkpoint). exp243 only has greedy pass@1; the
settings behind the earlier pass@5 numbers were not recorded, so every row is re-measured here:
  Dense | MoEfication (weight k-means grouping) | G-MoEfication (co-activation grouping) | ResMoE
All routed rows use the same x-router; the baselines use the mean representative and no shared floor;
ResMoE = keep-pattern grouping + shared floor + rank-128 correction with validation gating (a layer's
correction is kept only if its in-situ R2 on N_VAL held-out tokens is > 0).
Protocol: n samples per problem (temperature TEMP, top_p TOP_P, max MAXNEW new tokens, the MultiPL-E stop
tokens), generation seeded per problem so every row sees the same seeds; pass@k is the unbiased estimator
1 - C(n-c,k)/C(n,k) averaged over the 158 problems, with a bootstrap 95% CI over problems. Greedy pass@1
and code perplexity (mbpp held-out, as exp243) are reported alongside.
Run: HHMODEL=ckpts/santacoder-java python3 experiments/256_santacoder_java_passk.py   (needs javac/java on PATH)
Env: KEEPS (0.85,0.50) ppl keeps, PASSK_KEEPS (0.50) keeps that also get pass@k, N (10), TEMP (0.2),
     TOP_P (0.95), MAXNEW (350), SEED (0), JOBS (16) parallel javac/java, OUT=json path.
     CALIB=mbpp (exp243 protocol: Python calibration / ppl text) | java (Java blocks of the exp251 fine-tuning
     corpus, i.e. in-domain for MultiPL-E Java; its ppl is on text the model was fine-tuned on).
     ROWS (dense,moef,gmoe,resmoe); kp / kpsf add the ResMoE build-up without correction (keep-pattern
     grouping, + shared floor).
"""
from __future__ import annotations
import sys, pathlib, gc, os, types, math, json, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", str(pathlib.Path(__file__).resolve().parents[1] / "ckpts" / "santacoder-java"))
N_CALIB = int(os.environ.get("NCALIB", "8192"))
N_EVAL = int(os.environ.get("NEVAL", "4096"))
N_VAL = 1024                                                # gating tokens, disjoint from calibration and eval
CHUNK = 512
K = 128
RCORR = 128
RFEAT = 512
SHARED = float(os.environ.get("SHARED", "0.5"))             # ResMoE: fraction of the budget as always-on shared floor
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.85,0.50").split(",")]
PASSK_KEEPS = [float(x) for x in os.environ.get("PASSK_KEEPS", "0.50").split(",")]
N = int(os.environ.get("N", "10"))
TEMP = float(os.environ.get("TEMP", "0.2"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAXNEW = int(os.environ.get("MAXNEW", "350"))
SEED = int(os.environ.get("SEED", "0"))
JOBS = int(os.environ.get("JOBS", "16"))
OUT = os.environ.get("OUT", "")
CALIB = os.environ.get("CALIB", "mbpp")
ROWS = os.environ.get("ROWS", "dense,moef,gmoe,resmoe").split(",")
SPEC = dict(moef=('MoEfication', 'weight', False, False), gmoe=('G-MoEfication', 'coact', False, False),
            kp=('keep-pattern', 'keep', False, False), kpsf=('keep-pat+SF', 'keep', True, False),
            resmoe=('ResMoE', 'keep', True, True))        # name, grouping, shared floor, correction
KS = [k for k in (1, 5, 10) if k <= N]
ROOT = pathlib.Path(__file__).resolve().parents[1]
JAR = str(ROOT / "mple_java" / "lib" / "javatuples-1.2.jar")
WD = str(ROOT / "mple_java")


class Stop(Exception):
    pass


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
    torch.manual_seed(SEED)                                 # generation reseeds the global RNG; keep init fixed
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(SEED)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def keep_topB_neuron(score, B):
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def keep_topB_group(score_g, gsz, gf, B):
    N_, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N_, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, gf]


def fwd(mlp, x):
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    a = mlp.act(mlp.c_fc(xf))
    cfg = mlp._cfg
    if cfg['mode'] == 'dense':
        return mlp.c_proj(a).reshape(sh)
    z = (xf.float() - mlp._xbar) @ mlp._P
    sg = mlp._router(z)                                     # deployable x-router selection
    if mlp._sharedg is not None:
        sg[:, mlp._sharedg] = float('inf')                  # always-on shared-floor groups
    m = keep_topB_group(sg, mlp._gsz, mlp._gf, mlp._B).to(a.dtype)
    kept = a * m + mlp._abar.to(a.dtype) * (1 - m)
    out = mlp.c_proj(kept)
    meas = cfg.get('meas')
    if cfg['mode'] == 'routecorr' and (cfg.get('corr_on', True) or meas is not None):
        ehat = mlp._pred(z) @ mlp._Br.T
        if meas is not None:                                # validation gating: in-situ fit of this layer's correction
            E = (a.float() - kept.float()) @ mlp._Wd.T
            meas[0] += float(((E - ehat) ** 2).sum()); meas[1] += float((E ** 2).sum())
            raise Stop
        out = out + ehat.to(out.dtype)
    return out.reshape(sh)


def run_java(prog):
    d = tempfile.mkdtemp(dir=WD); jf = os.path.join(d, "Problem.java")
    with open(jf, 'w') as f:
        f.write(prog)
    ok = False
    try:
        c = subprocess.run(["javac", "-cp", JAR, "-d", d, jf], capture_output=True, timeout=60)
        if c.returncode == 0:
            r = subprocess.run(["java", "-ea", "-cp", d + ":" + JAR, "Problem"], capture_output=True, timeout=15)
            ok = r.returncode == 0
    except Exception:
        pass
    subprocess.run(["rm", "-rf", d])
    return ok


def pass_at_k(c, n, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def passk_stats(correct, n):
    """correct: per-problem #passing samples -> {k: (pass@k, ci_lo, ci_hi)}, bootstrap over problems."""
    g = torch.Generator().manual_seed(0); P = len(correct)
    idx = torch.randint(0, P, (5000, P), generator=g)
    out = {}
    for k in KS:
        v = torch.tensor([pass_at_k(c, n, k) for c in correct], dtype=torch.float64)
        bs = v[idx].mean(1).sort().values
        out[k] = (float(v.mean()), float(bs[int(0.025 * 5000)]), float(bs[int(0.975 * 5000)]))
    return out


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = []
    if CALIB == "java":                                     # Java blocks of the fine-tuning corpus (as exp251)
        import re
        java_re = re.compile(r"```java\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        for ex in load_dataset("rombodawg/MegaCodeTraining", split="train"):
            code += [b.strip() for b in java_re.findall(ex.get("ASSISTANT") or "") if len(b.strip()) > 40]
            if len(code) >= 400:
                break
    else:
        for sp in ["train", "test", "validation", "prompt"]:
            try:
                code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
            except Exception:
                pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    assert len(ids) >= N_CALIB + N_EVAL + N_VAL, len(ids)
    problems = load_dataset('nuprl/MultiPL-E', 'humaneval-java', split='test')
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.transformer.h; nL = len(layers)
    torch.set_grad_enabled(False)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0

    def set_mode(mode, **kw):
        for li in range(nL):
            layers[li].mlp._cfg = dict(mode=mode, **kw)

    def generate_eval(tag, n):
        """n=1 -> greedy; n>1 -> n seeded samples per problem. returns per-problem #passing samples."""
        model.config.use_cache = True
        kw = dict(do_sample=True, temperature=TEMP, top_p=TOP_P, num_return_sequences=n) if n > 1 else dict(do_sample=False)
        futs = []
        with ThreadPoolExecutor(JOBS) as pool:
            for pi, ex in enumerate(problems):
                iin = tok(ex['prompt'], return_tensors='pt').input_ids.to(dev)
                torch.manual_seed(SEED * 100003 + pi)       # same sampling seed for every row
                out = model.generate(iin, max_new_tokens=MAXNEW, pad_token_id=eos, stop_strings=ex['stop_tokens'],
                                     tokenizer=tok, **kw)
                progs = []
                for row in out:
                    gen = tok.decode(row[iin.shape[1]:], skip_special_tokens=True); cut = len(gen)
                    for st in ex['stop_tokens']:
                        j = gen.find(st)
                        if j != -1:
                            cut = min(cut, j)
                    progs.append(ex['prompt'] + gen[:cut] + "\n" + ex['tests'])
                futs.append([pool.submit(run_java, p) for p in progs])
            correct = [sum(f.result() for f in fs) for fs in futs]
        model.config.use_cache = False
        return correct

    def ce_eval():                                          # causal-LM perplexity on held-out code
        tot = 0.0; n = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); n += tgt.numel()
        return math.exp(tot / n)

    RES = dict(model=MODEL, calib=CALIB, n=N, temp=TEMP, top_p=TOP_P, maxnew=MAXNEW, seed=SEED, problems=len(problems), rows={})

    def evaluate(row, bf, extra=None):
        r = RES['rows'].setdefault(row, {}); kp = f"{bf}"
        r[kp] = dict(ppl=ce_eval(), **(extra or {}))
        msg = f"  [{row:>13} keep{int(bf*100):>3}] ppl {r[kp]['ppl']:.3f}"
        if bf in PASSK_KEEPS or row == 'Dense':
            g = generate_eval(row, 1); c = generate_eval(row, N); st = passk_stats(c, N)
            r[kp].update(greedy_pass1=sum(g) / len(g), greedy=g, correct=c,
                         **{f"pass@{k}": st[k][0] for k in KS}, **{f"pass@{k}_ci95": st[k][1:] for k in KS})
            msg += f" | greedy pass@1 {sum(g)/len(g):.4f} | " + " | ".join(
                f"pass@{k} {st[k][0]:.4f} [{st[k][1]:.3f},{st[k][2]:.3f}]" for k in KS)
        print(msg + (f" | {extra}" if extra else ""), flush=True)

    # ---- harvest ----
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        mlp = layers[li].mlp
        hs.append(mlp.c_fc.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(mlp.c_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"  MODEL={MODEL} dff={dff} nL={nL} (GeLU decoder) | calib={CALIB} ({len(ids)} tok) | n={N} T={TEMP} top_p={TOP_P} max_new={MAXNEW} seed={SEED} "
          f"| {len(problems)} problems", flush=True)

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev); x = torch.cat(capx[li]).float().to(dev)
        Wd = layers[li].mlp.c_proj.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        Wi = layers[li].mlp.c_fc.weight.detach().float(); Wi = Wi if Wi.shape[0] == dff else Wi.T
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a.half().cpu(), x=x.half().cpu(), abar=a.mean(0), vn=Wd.norm(dim=0), Wd=Wd.cpu(), Wi=Wi.cpu(),
                       xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del a, x, Wd; gc.collect(); torch.cuda.empty_cache()
    for li in range(nL):
        layers[li].mlp.forward = types.MethodType(fwd, layers[li].mlp)
    set_mode('dense')

    def grouping(li, bf, kind):
        s = STR[li]; a = s['a'].float().to(dev); vn = s['vn']; abar = s['abar']; B = int(round(bf * dff))
        w = ((a - abar).abs() * vn).mean(0)
        if kind == 'weight':                                # MoEfication: balanced k-means on input-weight rows
            Wi = s['Wi'].to(dev); gl = balanced_assign(Wi, kmeans_centroids(Wi, K, seed=0))
        else:
            if kind == 'coact':                             # G-MoEfication: co-activation profile
                feat = (a - abar).T.contiguous(); feat = feat / feat.norm(dim=1, keepdim=True).clamp(min=1e-6)
            else:                                           # ResMoE: per-token co-keep pattern
                feat = keep_topB_neuron((a - abar).abs() * vn, B).float().T.contiguous()
            gl = balanced_assign(feat, weighted_kmeans_centroids(feat, w, K, seed=0)); del feat
        gsz = torch.bincount(gl, minlength=K).float()
        del a; gc.collect(); torch.cuda.empty_cache()
        return gsz, gl

    def group_scores(li, gsz, gf):
        s = STR[li]; a = s['a'].float().to(dev)
        return torch.zeros(a.shape[0], K, device=dev).index_add_(1, gf, ((a - s['abar']) * s['vn']) ** 2)

    def install(li, bf, gsz, gf, shared):
        """grouping + x-router (per-group oracle score from PCA(x)) [+ shared floor] on layer li."""
        mlp = layers[li].mlp; s = STR[li]; B = int(round(bf * dff)); sg = group_scores(li, gsz, gf)
        mlp._abar = s['abar']; mlp._gsz = gsz; mlp._gf = gf; mlp._B = B; mlp._xbar = s['xbar']; mlp._P = s['P']
        mlp._sharedg = None
        if shared:                                          # always-on top groups by mean contribution (~SHARED*B neurons)
            order = sg.mean(0).argsort(descending=True)
            nsh = int((gsz[order].cumsum(0) < int(round(SHARED * B))).sum().item()) + 1
            mlp._sharedg = torch.zeros(K, dtype=torch.bool, device=dev); mlp._sharedg[order[:nsh]] = True
        mlp._router = mlp_fit((s['x'].float().to(dev) - s['xbar']) @ s['P'], sg, dev)
        del sg; gc.collect(); torch.cuda.empty_cache()

    def fit_correction(li):
        """rank-r basis + predictor of the deployed (router + shared floor) selection's own dropped-output error."""
        mlp = layers[li].mlp; s = STR[li]; a = s['a'].float().to(dev); Wd = s['Wd'].to(dev)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        sg = mlp._router(z)
        if mlp._sharedg is not None:
            sg[:, mlp._sharedg] = float('inf')
        m = keep_topB_group(sg, mlp._gsz, mlp._gf, mlp._B).to(a.dtype)
        E = (a - (a * m + s['abar'] * (1 - m))) @ Wd.T
        _, _, Vt = torch.linalg.svd(E, full_matrices=False); mlp._Br = Vt[:RCORR].T
        mlp._pred = mlp_fit(z, E @ mlp._Br, dev)
        del a, Wd, z, sg, m, E, Vt; gc.collect(); torch.cuda.empty_cache()

    def gate():
        """greedy validation gating: layer li is judged with layers < li as already decided."""
        set_mode('routecorr', corr_on=False); dropped = []
        for li in range(nL):
            mlp = layers[li].mlp; mlp._Wd = STR[li]['Wd'].to(dev); mlp._cfg['meas'] = [0.0, 0.0]
            for c0 in range(N_CALIB + N_EVAL, N_CALIB + N_EVAL + N_VAL, CHUNK):
                try:
                    model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
                except Stop:
                    pass
            num, den = mlp._cfg.pop('meas'); mlp._Wd = None
            mlp._cfg['corr_on'] = num < den
            if not mlp._cfg['corr_on']:
                dropped.append(li)
        return dropped

    print(f"\n  SantaCoder MultiPL-E Java, all rows under one protocol (x-router for every routed row)\n", flush=True)
    if 'dense' in ROWS:
        evaluate('Dense', 1.0)
    for bf in KEEPS:
        for row in [r for r in ROWS if r != 'dense']:
            name, kind, shared, corr = SPEC[row]
            for li in range(nL):
                install(li, bf, *grouping(li, bf, kind), shared=shared)
                if corr:
                    fit_correction(li)
            if corr:
                dropped = gate()                            # leaves mode='routecorr' with the gated corr_on flags
                evaluate(name, bf, extra=dict(gate_dropped=dropped))
            else:
                set_mode('route'); evaluate(name, bf)
            set_mode('dense')
        if OUT:
            pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(OUT).write_text(json.dumps(RES, indent=1))
    if OUT:
        print(f"\n  wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
