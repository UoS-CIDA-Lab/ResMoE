"""Experiment 257 — HumanEval pass@1 with UNCERTAINTY, for the oracle rows of exp232 and the deployable
x-router pipeline of exp230 (Qwen-Coder, keep-pattern K=128, keep50, rank-128 correction). exp232 reports one
run per row at group-ORACLE selection; here each row is repeated over conversion seeds (router / correction
MLP init and batches; the grouping is fixed) and given a bootstrap CI over the 164 problems:
  dense | neuron-oracle (ceiling)                                   deterministic, evaluated once
  group-oracle (floor) | group-oracle + correction                  the exp232 rows
  x-router (floor)     | x-router + correction, validation-gated    the deployable pipeline (exp230 + gating)
  x-router + correction on all layers                               only when gating dropped a layer
Protocol = lm-eval's `humaneval` task (as exp232): prompt -> greedy completion, stop at \\nclass \\ndef \\n# \\nif
\\nprint, max 1024 new tokens, program = prompt + completion + test + check(entry_point). Generation is batched
(left padding) and the programs are run in guarded subprocesses, so no lm-eval install is needed.
Reported: per-seed pass@1, mean ± std over seeds, 95% bootstrap CI over problems of the seed-averaged pass
rate, and a PAIRED bootstrap CI of each correction's gain over its own floor.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/257_humaneval_uncertainty.py
Env: SEEDS (0,1,2,3,4), KEEP (0.50), BATCH (16), MAXNEW (1024), JOBS (8), OUT=json path.
"""
from __future__ import annotations
import sys, pathlib, gc, os, json, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.reproducibility import load_mbpp_code
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_VAL = 1024                                                # gating tokens, disjoint from calibration
VAL0 = N_CALIB + 1024
CHUNK = 512
K = 128
RCORR = 128
BF = float(os.environ.get("KEEP", "0.50"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
BATCH = int(os.environ.get("BATCH", "16"))
MAXNEW = int(os.environ.get("MAXNEW", "1024"))
JOBS = int(os.environ.get("JOBS", "8"))
OUT = os.environ.get("OUT", "")
UNTIL = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
GUARD = """import os, shutil, subprocess, resource, faulthandler
faulthandler.disable()
resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
for _m, _ns in ((os, 'kill system remove removedirs rmdir unlink rename renames truncate fork forkpty killpg chmod chown putenv'),
                (shutil, 'rmtree move chown'), (subprocess, 'Popen')):
    for _n in _ns.split():
        setattr(_m, _n, None)
"""


class Stop(Exception):
    pass


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


def mlp_fit(X, Y, dev, seed, steps=3000, hidden=512, lr=3e-3, bs=2048):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(seed)
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


def run_py(prog):
    d = tempfile.mkdtemp(); f = os.path.join(d, "prog.py")
    with open(f, "w") as fh:
        fh.write(GUARD + prog)
    try:
        ok = subprocess.run([sys.executable, "-I", f], capture_output=True, timeout=10, cwd=d,
                            env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}).returncode == 0
    except Exception:
        ok = False
    subprocess.run(["rm", "-rf", d])
    return ok


def boot_ci(v, n=10000):
    """v: [problems] values -> 95% bootstrap CI of the mean over problems."""
    g = torch.Generator().manual_seed(0); v = torch.as_tensor(v, dtype=torch.float64)
    bs = v[torch.randint(0, len(v), (n, len(v)), generator=g)].mean(1).sort().values
    return float(bs[int(0.025 * n)]), float(bs[int(0.975 * n)])


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True); tok.padding_side = "left"
    code = load_mbpp_code()
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    problems = list(load_dataset("openai/openai_humaneval", split="test"))
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    torch.set_grad_enabled(False)

    STR = {}
    RUN = dict(mode=None, selr=None, corr=None, on=(), gate=None, meas=None, E=None)
    XC = {li: None for li in range(nL)}

    def mask(mode, li, a, x):
        s = STR[li]
        if mode == 'neuron':
            return keep_topB_neuron((a - s['abar']).abs() * s['vn'], B)
        if mode == 'oracle':
            sg = torch.zeros(a.shape[0], K, device=a.device).index_add_(1, s['gf'], ((a - s['abar']) * s['vn']) ** 2)
        else:                                               # x-router: MLP on PCA(x) -> per-group residual norm
            sg = RUN['selr'][li]((x - s['xbar']) @ s['P']).float()
        return keep_topB_group(sg, s['gsz'], s['gf'], B)

    def gpre(li):
        def hook(_m, args):
            XC[li] = args[0].reshape(-1, args[0].shape[-1]).float()
        return hook

    def dpre(li):
        def hook(_m, args):
            if RUN['mode'] is None:
                return None
            a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1]); s = STR[li]
            m = mask(RUN['mode'], li, af.float(), XC[li])
            if RUN['gate'] == li:                           # true dropped-output error on the in-situ input
                RUN['E'] = ((af.float() - s['abar']) * ~m) @ s['Wd'].T
            return (torch.where(m, af, s['abar'].to(a.dtype)).reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            gating = RUN['gate'] == li
            if RUN['mode'] is None or RUN['corr'] is None or not (li in RUN['on'] or gating):
                return None
            s = STR[li]; Br, pred = RUN['corr'][li]; sh = output.shape
            ehat = pred((XC[li] - s['xbar']) @ s['P']) @ Br.T
            if gating:
                RUN['meas'][0] += float(((RUN['E'] - ehat) ** 2).sum()); RUN['meas'][1] += float((RUN['E'] ** 2).sum())
                raise Stop
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    order = sorted(range(len(problems)), key=lambda i: -len(tok(problems[i]['prompt']).input_ids))

    def humaneval(tag, mode=None, selr=None, corr=None, on=()):
        """greedy completion of every problem under the given configuration -> per-problem pass (0/1)."""
        RUN.update(mode=mode, selr=selr, corr=corr, on=set(on), gate=None)
        model.config.use_cache = True; futs = {}
        with ThreadPoolExecutor(JOBS) as pool:
            for b0 in range(0, len(order), BATCH):
                idx = order[b0:b0 + BATCH]
                enc = tok([problems[i]['prompt'] for i in idx], return_tensors="pt", padding=True).to(dev)
                out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, stop_strings=UNTIL, tokenizer=tok,
                                     pad_token_id=tok.pad_token_id)
                for j, i in enumerate(idx):
                    gen = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
                    for u in UNTIL:
                        gen = gen.split(u)[0]
                    p = problems[i]
                    futs[i] = pool.submit(run_py, p['prompt'] + gen + "\n" + p['test'] + "\n" + f"check({p['entry_point']})")
            passed = [int(futs[i].result()) for i in range(len(problems))]
        model.config.use_cache = False; RUN['mode'] = None
        print(f"  [{tag}] HumanEval pass@1 = {sum(passed)/len(passed):.4f} ({sum(passed)}/{len(passed)})", flush=True)
        return passed

    # ---- harvest calibration activations, statistics, keep-pattern grouping (fixed across seeds) ----
    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(downs[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(gproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]; B = int(round(BF * dff))
    print(f"  MODEL={MODEL} dff={dff} layers={nL} keep{int(BF*100)} K={K} r={RCORR} seeds={SEEDS} | "
          f"{len(problems)} problems, greedy, max_new={MAXNEW}, batch={BATCH}", flush=True)
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float(); vn = Wd.norm(dim=0); abar = a.mean(0)
        con = (a - abar).abs() * vn; Bind = keep_topB_neuron(con, B).float()
        gf = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), con.mean(0), K, seed=0))
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a.half().cpu(), x=x.half().cpu(), Wd=Wd, vn=vn, abar=abar, xbar=xbar, P=VtX[:512].T,
                       gf=gf, gsz=torch.bincount(gf, minlength=K).float())
        capa[li] = None; capx[li] = None
        del a, x, con, Bind, VtX; gc.collect(); torch.cuda.empty_cache()

    def convert(seed):
        """seed-dependent parts: x-router and the two corrections (trained on each selector's own error)."""
        selr = {}; corr = {'oracle': {}, 'xrouter': {}}
        for li in range(nL):
            s = STR[li]; a = s['a'].float().to(dev); x = s['x'].float().to(dev); z = (x - s['xbar']) @ s['P']
            dev_a = a - s['abar']; rn = torch.zeros(a.shape[0], K, device=dev)
            for g in range(K):
                ix = (s['gf'] == g).nonzero().flatten()
                if len(ix):
                    rn[:, g] = (dev_a[:, ix] @ s['Wd'][:, ix].T).norm(dim=1)
            selr[li] = mlp_fit(z, rn, dev, seed); RUN['selr'] = selr
            for sel in corr:
                E = (dev_a * ~mask(sel, li, a, x)) @ s['Wd'].T
                _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
                corr[sel][li] = (Br, mlp_fit(z, E @ Br, dev, seed))
            del a, x, z, dev_a, rn, E, Vt; gc.collect(); torch.cuda.empty_cache()
        return selr, corr

    def gate(selr, corr):
        """greedy validation gating of the x-router correction: keep layer li iff its in-situ R2 > 0."""
        on = []
        for li in range(nL):
            RUN.update(mode='xrouter', selr=selr, corr=corr, on=set(on), gate=li, meas=[0.0, 0.0])
            for c0 in range(VAL0, VAL0 + N_VAL, CHUNK):
                try:
                    model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
                except Stop:
                    pass
            if RUN['meas'][0] < RUN['meas'][1]:
                on.append(li)
        RUN.update(mode=None, gate=None)
        return on

    RES = dict(model=MODEL, keep=BF, K=K, rcorr=RCORR, seeds=SEEDS, problems=len(problems), maxnew=MAXNEW,
               rows={}, gate_dropped={})
    R = RES['rows']

    def save():
        if OUT:
            pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(OUT).write_text(json.dumps(RES))

    R['dense'] = [humaneval("dense")]
    R['neuron-oracle'] = [humaneval("neuron-oracle (ceiling)", 'neuron')]
    R['oracle floor'] = [humaneval("group-oracle (floor)", 'oracle')]; save()
    for seed in SEEDS:
        selr, corr = convert(seed); on = gate(selr, corr['xrouter'])
        dropped = [li for li in range(nL) if li not in on]; RES['gate_dropped'][str(seed)] = dropped
        print(f"  seed {seed}: converted; gating dropped layers {dropped}", flush=True)
        R.setdefault('oracle + corr', []).append(humaneval(f"seed {seed} group-oracle + corr", 'oracle', corr=corr['oracle'], on=range(nL)))
        R.setdefault('x-router floor', []).append(humaneval(f"seed {seed} x-router (floor)", 'xrouter', selr))
        R.setdefault('x-router + corr (gated)', []).append(
            humaneval(f"seed {seed} x-router + corr (gated)", 'xrouter', selr, corr['xrouter'], on))
        if dropped:
            R.setdefault('x-router + corr (all layers)', []).append(
                humaneval(f"seed {seed} x-router + corr (all layers)", 'xrouter', selr, corr['xrouter'], range(nL)))
        del selr, corr; gc.collect(); torch.cuda.empty_cache(); save()

    def avg(row):                                           # per-problem pass rate averaged over seeds
        return torch.tensor(R[row], dtype=torch.float64).mean(0)

    print(f"\n  HumanEval pass@1, keep{int(BF*100)} (mean ± std over seeds; 95% bootstrap CI over {len(problems)} problems)", flush=True)
    RES['summary'] = {}
    for row in R:
        per = [sum(v) / len(v) for v in R[row]]; m = sum(per) / len(per)
        sd = (sum((p - m) ** 2 for p in per) / len(per)) ** 0.5
        lo, hi = boot_ci(avg(row)); RES['summary'][row] = dict(mean=m, std=sd, ci95=[lo, hi], per_seed=per)
        print(f"  {row:<30} | {m:.3f} ± {sd:.3f} | CI [{lo:.3f}, {hi:.3f}] | per seed " + " ".join(f"{p:.3f}" for p in per), flush=True)
    RES['paired'] = {}
    for hi_row, lo_row in (('oracle + corr', 'oracle floor'), ('x-router + corr (gated)', 'x-router floor'),
                           ('x-router + corr (all layers)', 'x-router floor')):
        if hi_row in R:
            d = avg(hi_row) - avg(lo_row); lo, hi = boot_ci(d)
            RES['paired'][f"{hi_row} - {lo_row}"] = dict(gain=float(d.mean()), ci95=[lo, hi])
            print(f"  gain {hi_row} over {lo_row}: {float(d.mean()):+.3f}  paired CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
    save()
    if OUT:
        print(f"  wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
