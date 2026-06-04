"""Experiment 178 — DEPLOYABLE adaptive-budget (shared-floor): phase-1 deployability gate.
exp176 (ORACLE) found shared-floor + per-token routed-budget concentration beats uniform (keep0.5
sf0.6: 28.55 vs 32.14, +11%). Is that gain LEARNABLE? Train a router to predict per-routed-group
residual; select either UNIFORM (per-token fixed budget) or ADAPTIVE (global threshold tau on
PREDICTED score => variable per-token budget). Compare deploy_uniform vs deploy_thresh (+ oracle
refs) at matched compute. deploy adaptive < deploy uniform => the per-token budget signal is
predictable from x => oracle gain survives deployability => proceed to distill. Qwen-0.5B.
Run: python3 experiments/178_deployable_adaptive_budget.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "bigcode/gpt_bigcode-santacoder")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
KROUTE = 64


def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a


def mlp_fit(X, Y, dev, steps=800, lr=3e-3, bs=2048, hidden=128):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(0)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def gm_forward(mlp, x):
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    a = mlp.act(mlp.c_fc(xf))                           # GeLU FFN hidden (SantaCoder)
    N = a.shape[0]; Kc = mlp._gsz.shape[0]
    src, sel = mlp._mode.split("_")                      # src: oracle|deploy ; sel: uniform|global|thresh
    if src == "oracle":
        sq = ((a.float() - mlp._mean.float()).abs() * mlp._vn) ** 2
        score = torch.zeros(N, Kc, device=a.device).index_add_(1, mlp._routed_grp_full, sq)
    else:
        z = (xf.float() - mlp._xbar) @ mlp._P
        score = mlp._router(z).float()
    if sel == "uniform":                                 # per-token fixed routed budget (top route_budget)
        order = score.argsort(1, descending=True); so = mlp._gsz[order]
        keep_ord = (so.cumsum(1) - so) < mlp._route_budget
        selg = torch.zeros(N, Kc, dtype=torch.bool, device=a.device).scatter_(1, order, keep_ord)
    elif sel == "global":                                # exact global top-k across all (token,group) cells
        cost = mlp._gsz.float().unsqueeze(0).expand(N, Kc).reshape(-1)
        o = score.reshape(-1).argsort(descending=True)
        cum = cost[o].cumsum(0) - cost[o]
        kf = torch.zeros(N * Kc, dtype=torch.bool, device=a.device); kf[o] = cum < (N * mlp._route_budget)
        selg = kf.reshape(N, Kc)
    elif sel == "thresh":                                # CAUSAL adaptive: per-token score >= calibrated tau
        selg = score >= mlp._tau
    else:                                                # CAUSAL adaptive: per-token top-p (cumulative fraction)
        order = score.argsort(1, descending=True)
        ssort = score.gather(1, order).clamp_min(0)
        cums = ssort.cumsum(1); tot = cums[:, -1:].clamp_min(1e-9)
        keep_ord = (cums - ssort) < (mlp._topp * tot)    # keep groups until cumulative >= p*total
        selg = torch.zeros(N, Kc, dtype=torch.bool, device=a.device).scatter_(1, order, keep_ord)
    m = mlp._shared_mask.expand(N, -1).clone()
    m = m + selg[:, mlp._routed_grp_full].to(m.dtype) * mlp._routed_is
    m = m.clamp(max=1.0)
    mlp._kept = m.mean().item()
    out = mlp.c_proj(a * m + mlp._rep * (1 - m))
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = []                                                # CODE corpus (mbpp) — domain-matched to coder models
    for sp in ["train", "test", "validation", "prompt"]:
        try:
            code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    print(f"  MODEL={MODEL}  code tokens={ids.numel()}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.transformer.h; nL = len(layers)

    @torch.no_grad()
    def ce_eval():
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return tot / ntok

    JAR = str(pathlib.Path(__file__).resolve().parents[1] / "mple_java" / "lib" / "javatuples-1.2.jar")
    WD = str(pathlib.Path(__file__).resolve().parents[1] / "mple_java")

    def mple_java(tag, max_new=350):
        import subprocess, tempfile, os as _os
        from datasets import load_dataset as _ld
        ds = _ld('nuprl/MultiPL-E', 'humaneval-java', split='test', trust_remote_code=True)
        torch.set_grad_enabled(False); model.config.use_cache = True
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

    print(f"\n  SantaCoder MultiPL-E Java REPRODUCTION (paper: G-MoE 13.77%, oracle 17.81% @keep85)\n", flush=True)
    print("  [DENSE (unpatched)] MultiPL-E Java pass@1 = 0.1519 (24/158) [prior run]", flush=True)

    torch.set_grad_enabled(False)
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        mlp = layers[li].mlp
        hs.append(mlp.c_fc.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))(li)))
        hs.append(mlp.c_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"Qwen2.5-0.5B: {nL} layers, dff={dff}", flush=True)
    LST = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        Wup = layers[li].mlp.c_fc.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        Wd = layers[li].mlp.c_proj.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        abar = a.mean(0); dev_a = a - abar
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        P = Vt[:min(512, Vt.shape[0])].T                       # up to 512 PCA dims
        contrib = dev_a.abs() * Wd.norm(dim=0)
        LST[li] = dict(Wup=Wup.cpu(), Wd=Wd.cpu(), abar=abar, dev_a=dev_a.cpu(), contrib=contrib.cpu(),
                       Z=((x - xbar) @ P).float().cpu(), xbar=xbar.float(), P=P.float())
        layers[li].mlp._mean = abar.to(model.dtype)
        del x, a, dev_a, contrib, Wup, Wd; gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0:
            print(f"  prepped {li}", flush=True)
    tr = slice(0, N_CALIB)

    def configure(bf, sf, strong=False):
        B = int(round(bf * dff))
        rfeat = 512 if strong else 128
        hidden = 512 if strong else 128
        steps = 2500 if strong else 800
        for li in range(nL):
            s = LST[li]
            Wup = s['Wup'].to(dev); Wd = s['Wd'].to(dev); dev_a = s['dev_a'].to(dev); contrib = s['contrib'].to(dev)
            topB = contrib.argsort(1, descending=True)[:, :B]
            freq = torch.zeros(dff, device=dev); freq.scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev))
            n_shared = int(round(sf * B))
            shared_idx = freq.topk(n_shared).indices if n_shared > 0 else torch.tensor([], dtype=torch.long, device=dev)
            shared_mask = torch.zeros(dff, device=dev); shared_mask[shared_idx] = 1.0
            routed_pool = (shared_mask < 0.5).nonzero().flatten()
            route_budget = B - n_shared
            grp_local = kmeans(F.normalize(Wup[routed_pool], dim=1), min(KROUTE, len(routed_pool)), seed=0)
            Kc = int(grp_local.max().item()) + 1
            gsz = torch.zeros(Kc, device=dev); grp_full = torch.zeros(dff, dtype=torch.long, device=dev)
            routed_is = torch.zeros(dff, device=dev); rn = torch.zeros(dev_a.shape[0], Kc, device=dev)
            for g in range(Kc):
                ix = routed_pool[(grp_local == g).nonzero().flatten()]
                gsz[g] = len(ix); grp_full[ix] = g; routed_is[ix] = 1.0
                if len(ix):
                    rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
            rf = min(rfeat, s['P'].shape[1])
            Ztr = s['Z'][:, :rf].to(dev)
            router = mlp_fit(Ztr[tr], rn[tr], dev, hidden=hidden, steps=steps)
            vn = Wd.norm(dim=0).clone()
            # calibrate tau: global threshold on PREDICTED score to hit avg route_budget neurons/token
            with torch.no_grad():
                pred = router(Ztr[tr]).float()                  # [Ntr, Kc]
            fc = gsz.float().unsqueeze(0).expand_as(pred).reshape(-1)
            fp = pred.reshape(-1); o = fp.argsort(descending=True)
            cum = fc[o].cumsum(0) - fc[o]
            keepn = cum < (pred.shape[0] * route_budget)
            tau = fp[o][keepn].min() if keepn.any() else fp.max()
            # calibrate top-p: find p s.t. avg kept neurons ~ route_budget (binary search on calib preds)
            pc = pred.clamp_min(0); psort, pord = pc.sort(1, descending=True)
            gsort = gsz[pord]; pcums = psort.cumsum(1); ptot = pcums[:, -1:].clamp_min(1e-9)
            lo_p, hi_p = 0.0, 1.0
            for _ in range(20):
                mid = (lo_p + hi_p) / 2
                kept = (gsort * (((pcums - psort) < mid * ptot).float())).sum(1).mean()
                if kept > route_budget:
                    hi_p = mid
                else:
                    lo_p = mid
            topp = (lo_p + hi_p) / 2
            del Wup, Wd, dev_a, contrib, rn, Ztr, pred, pc, psort, pord, pcums; gc.collect(); torch.cuda.empty_cache()
            mlp = layers[li].mlp
            mlp._xbar = s['xbar']; mlp._P = s['P'][:, :rf].contiguous(); mlp._router = router
            mlp._gsz = gsz.to(model.dtype); mlp._route_budget = float(route_budget); mlp._tau = tau; mlp._topp = topp
            mlp._shared_mask = shared_mask.to(model.dtype).unsqueeze(0)
            mlp._routed_grp_full = grp_full; mlp._routed_is = routed_is.to(model.dtype)
            mlp._rep = mlp._mean; mlp._vn = vn; mlp._mode = "deploy_uniform"
        for li in range(nL):
            layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    def set_mode(mode):
        for li in range(nL):
            layers[li].mlp._mode = mode

    def ppl():
        return float(torch.tensor(ce_eval()).exp())

    def meankept():
        return sum(layers[li].mlp._kept for li in range(nL)) / nL

    # G-MoE reproduction at keep85 (paper: G-MoE 13.77%, oracle 17.81%)
    configure(0.85, 0.0, strong=True)                    # G-MoE = route-all (sf=0), saturated router
    set_mode("oracle_uniform"); mple_java("85% G-MoE-ORACLE (route-all)")
    set_mode("deploy_uniform"); mple_java("85% G-MoE (route-all, trained router)")
    # Ours at keep85 (head-to-head)
    configure(0.85, 0.6, strong=True)
    set_mode("deploy_thresh"); mple_java("85% Ours (shared+thresh-adaptive)")
    print("\nREAD: if our G-MoE pass@1 (oracle ~ paper 17.81%, deploy ~ 13.77%) reproduces the paper =>")
    print("our G-MoEfication implementation is FAITHFUL (validates the baseline). Then Ours vs G-MoE is credible.")


if __name__ == "__main__":
    main()
