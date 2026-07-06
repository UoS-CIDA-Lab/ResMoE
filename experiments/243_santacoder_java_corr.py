"""Experiment 243 — our low-rank correction on G-MoEfication's own benchmark (SantaCoder MultiPL-E Java).
Adds OUR method (keep-pattern grouping + predicted rank-128 correction) to the G-MoE comparison, at their
setting keep85 and at aggressive keep50. SantaCoder = GeLU decoder (transformer.h, mlp.c_fc/c_proj).
Modes (MultiPL-E Java pass@1): dense | group-oracle (no corr) | group-oracle + correction | neuron-oracle.
Run: python3 experiments/243_santacoder_java_corr.py
"""
from __future__ import annotations
import sys, pathlib, gc, os, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "bigcode/gpt_bigcode-santacoder")
N_CALIB = int(os.environ.get("NCALIB", "8192"))
CHUNK = 512
K = 128
RCORR = 128
RFEAT = 512
SHARED = float(os.environ.get("SHARED", "0.5"))   # fraction of budget as always-on shared floor (default ON)
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.85,0.50").split(",")]


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


def oracle_mask(a, abar, vn, gsz, gf, B, sharedg=None):
    per = ((a - abar) * vn) ** 2
    sg = torch.zeros(a.shape[0], gsz.shape[0], device=a.device).index_add_(1, gf, per)
    if sharedg is not None:
        sg[:, sharedg] = float('inf')               # always-on shared-floor groups
    return keep_topB_group(sg, gsz, gf, B).to(a.dtype)


def fwd(mlp, x):
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    a = mlp.act(mlp.c_fc(xf))
    cfg = mlp._cfg
    if cfg['mode'] == 'dense':
        return mlp.c_proj(a).reshape(sh)
    af = a.float()
    if cfg['mode'] == 'neuron':                           # per-neuron ceiling (no shared floor)
        m = keep_topB_neuron((af - mlp._abar).abs() * mlp._vn, mlp._B).to(a.dtype)
    elif cfg['mode'] in ('route', 'routecorr'):           # deployable x-router selection
        sg = mlp._router((xf.float() - mlp._xbar) @ mlp._P)
        if mlp._sharedg is not None:
            sg[:, mlp._sharedg] = float('inf')           # always-on shared-floor groups
        m = keep_topB_group(sg, mlp._gsz, mlp._gf, mlp._B).to(a.dtype)
    else:                                                 # oracle group selection
        m = oracle_mask(af, mlp._abar, mlp._vn, mlp._gsz, mlp._gf, mlp._B, mlp._sharedg).to(a.dtype)
    kept = a * m + mlp._abar.to(a.dtype) * (1 - m)
    out = mlp.c_proj(kept)
    if cfg['mode'] in ('corr', 'routecorr'):
        z = (xf.float() - mlp._xbar) @ mlp._P
        ehat = mlp._pred(z) @ mlp._Br.T
        out = out + ehat.to(out.dtype)
    return out.reshape(sh)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = []
    for sp in ["train", "test", "validation", "prompt"]:
        try:
            code += load_dataset("google-research-datasets/mbpp", "full", split=sp)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.transformer.h; nL = len(layers)
    torch.set_grad_enabled(False)

    JAR = str(pathlib.Path(__file__).resolve().parents[1] / "mple_java" / "lib" / "javatuples-1.2.jar")
    WD = str(pathlib.Path(__file__).resolve().parents[1] / "mple_java")

    def set_mode(mode):
        for li in range(nL):
            layers[li].mlp._cfg = {'mode': mode}

    def mple_java(tag, max_new=int(os.environ.get("MAXNEW", "350"))):
        import subprocess, tempfile, os as _os
        from datasets import load_dataset as _ld
        ds = _ld('nuprl/MultiPL-E', 'humaneval-java', split='test', trust_remote_code=True)
        model.config.use_cache = True
        passed = 0; total = 0
        eos = tok.eos_token_id if tok.eos_token_id is not None else 0
        for ex in ds:
            iin = tok(ex['prompt'], return_tensors='pt').input_ids.to(dev)
            out = model.generate(iin, max_new_tokens=max_new, do_sample=False, pad_token_id=eos,
                                 stop_strings=ex['stop_tokens'], tokenizer=tok)
            gen = tok.decode(out[0, iin.shape[1]:], skip_special_tokens=True)
            cut = len(gen)
            for st in ex['stop_tokens']:
                j = gen.find(st)
                if j != -1:
                    cut = min(cut, j)
            prog = ex['prompt'] + gen[:cut] + "\n" + ex['tests']
            d = tempfile.mkdtemp(dir=WD); jf = _os.path.join(d, "Problem.java"); open(jf, 'w').write(prog)
            total += 1
            try:
                c = subprocess.run(["javac", "-cp", JAR, "-d", d, jf], capture_output=True, timeout=40)
                if c.returncode == 0:
                    r = subprocess.run(["java", "-ea", "-cp", d + ":" + JAR, "Problem"], capture_output=True, timeout=15)
                    if r.returncode == 0:
                        passed += 1
            except Exception:
                pass
            subprocess.run(["rm", "-rf", d])
        model.config.use_cache = False
        print(f"  [{tag}] MultiPL-E Java pass@1 = {passed/total:.4f} ({passed}/{total})", flush=True)
        return passed / total

    import math
    PPL = os.environ.get("PPL", "0") == "1"
    N_EVAL = int(os.environ.get("NEVAL", "4096"))

    def ce_eval():                                   # causal-LM perplexity on held-out code
        tot = 0.0; n = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); n += tgt.numel()
        return math.exp(tot / n)

    def run_eval(tag):
        if PPL:
            r = ce_eval(); print(f"  [{tag}] ppl = {r:.4f}", flush=True); return r
        return mple_java(tag)

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
    print(f"  MODEL={MODEL} dff={dff} nL={nL} (GeLU decoder)", flush=True)

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev); x = torch.cat(capx[li]).float().to(dev)
        Wd = layers[li].mlp.c_proj.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a.half().cpu(), x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None
        del a, x, Wd; gc.collect(); torch.cuda.empty_cache()
    for li in range(nL):
        layers[li].mlp.forward = types.MethodType(fwd, layers[li].mlp)
        layers[li].mlp._cfg = {'mode': 'dense'}

    def grouping(li, bf, kind='keep'):
        s = STR[li]; a = s['a'].float().to(dev); vn = s['vn']; abar = s['abar']; B = int(round(bf * dff))
        w = ((a - abar).abs() * vn).mean(0)
        if kind == 'coact':
            # G-MoEfication/MoEfication-style construction: cluster neurons by their
            # (centered, normalized) co-activation profile across calibration tokens.
            feat = (a - abar).T.contiguous()                      # [dff, N]
            feat = feat / feat.norm(dim=1, keepdim=True).clamp(min=1e-6)
        else:
            # ResMoE construction: cluster by per-token co-keep pattern.
            feat = keep_topB_neuron((a - abar).abs() * vn, B).float().T.contiguous()
        gl = balanced_assign(feat, weighted_kmeans_centroids(feat, w, K, seed=0))
        gsz = torch.zeros(K, device=dev)
        for g in range(K):
            gsz[g] = (gl == g).sum()
        del a, feat; gc.collect(); torch.cuda.empty_cache()
        return gsz, gl

    def shared_groups(li, bf, gsz, gf):
        # group-level shared floor: always-on top groups by mean output contribution.
        # Our-method-only (the G-MoE baseline does NOT use it).
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; vn = s['vn']
        B = int(round(bf * dff)); S = int(round(SHARED * B))
        per = ((a - abar) * vn) ** 2
        sg = torch.zeros(a.shape[0], int(gsz.shape[0]), device=dev).index_add_(1, gf, per)
        order = sg.mean(0).argsort(descending=True)
        nsh = int((gsz[order].cumsum(0) < S).sum().item()) + 1     # groups until ~S neurons
        shg = torch.zeros(int(gsz.shape[0]), dtype=torch.bool, device=dev)
        shg[order[:nsh]] = True
        del a, per, sg; gc.collect(); torch.cuda.empty_cache()
        return shg

    def basis_and_pred_route(li, bf, gsz, gf, router, sharedg=None):
        # correction trained on the deployed (x-router [+ shared-floor]) selection's own error
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        sg = router(z)
        if sharedg is not None:
            sg[:, sharedg] = float('inf')
        m = keep_topB_group(sg, gsz, gf, B).to(a.dtype)
        drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T
        del a, m, drop, Wd; gc.collect(); torch.cuda.empty_cache()
        Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
        c = E @ Br
        pred = mlp_fit(z, c, dev)
        del E, Ec, Vt, c, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def fit_router(li, gsz, gf):
        # deployable x-router: predict per-group oracle score from x (PCA feats)
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; vn = s['vn']
        per = ((a - abar) * vn) ** 2
        sg = torch.zeros(a.shape[0], int(gsz.shape[0]), device=dev).index_add_(1, gf, per)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        router = mlp_fit(z, sg, dev)
        del a, per, sg, z; gc.collect(); torch.cuda.empty_cache()
        return router

    def setgrp(GRP, bf, shared=True):
        B = int(round(bf * dff))
        for li in range(nL):
            mlp = layers[li].mlp; s = STR[li]; gsz, gf = GRP[li]
            mlp._abar = s['abar']; mlp._vn = s['vn']; mlp._gsz = gsz; mlp._gf = gf; mlp._B = B
            mlp._sharedg = shared_groups(li, bf, gsz, gf) if shared else None
            mlp._xbar = s['xbar']; mlp._P = s['P']

    print("\n  SantaCoder MultiPL-E Java: G-MoE baseline (no shared floor) vs ResMoE\n", flush=True)
    set_mode('dense'); dense = run_eval("dense")
    if os.environ.get("DENSEONLY", "0") == "1":
        print(f"  ==> DENSE only (max_new={int(os.environ.get('MAXNEW','350'))}): {dense:.4f}", flush=True)
        return
    for bf in KEEPS:
        kp = int(bf * 100)
        # (1) G-MoE BASELINE: co-activation grouping + x-router + mean, NO shared floor
        GRPc = {li: grouping(li, bf, kind='coact') for li in range(nL)}
        setgrp(GRPc, bf, shared=False)
        for li in range(nL):
            layers[li].mlp._router = fit_router(li, *GRPc[li])
        set_mode('route'); gmoe = run_eval(f"keep{kp} G-MoE baseline (coact, no SF)")
        # ResMoE build-up on keep-pattern grouping (its own router)
        GRPk = {li: grouping(li, bf, kind='keep') for li in range(nL)}
        for li in range(nL):
            layers[li].mlp._router = fit_router(li, *GRPk[li])
        # (2) + keep-pattern construction (still no shared floor)
        setgrp(GRPk, bf, shared=False)
        set_mode('route'); kpat = run_eval(f"keep{kp} +keep-pattern construction (no SF)")
        # (3) + shared-floor selection (our default selection) -- toggle SF on, same router
        setgrp(GRPk, bf, shared=True)
        set_mode('route'); sf = run_eval(f"keep{kp} +shared-floor selection")
        # (4) + correction = ResMoE (trained on the SF+router selection's own error)
        for li in range(nL):
            Br, pred = basis_and_pred_route(li, bf, *GRPk[li], layers[li].mlp._router,
                                            layers[li].mlp._sharedg)
            layers[li].mlp._Br = Br; layers[li].mlp._pred = pred
        set_mode('routecorr'); resmoe = run_eval(f"keep{kp} ResMoE (+correction)")
        print(f"  ==> keep{kp}: dense {dense:.4f} | G-MoE {gmoe:.4f} | +keep-pattern {kpat:.4f} | "
              f"+shared-floor {sf:.4f} | ResMoE {resmoe:.4f}\n", flush=True)
    print("READ: G-MoE baseline (NO shared floor) vs ResMoE build-up: +keep-pattern construction ->", flush=True)
    print("+shared-floor selection -> +rank-128 correction (=ResMoE). dense = upper bound.", flush=True)


if __name__ == "__main__":
    main()
