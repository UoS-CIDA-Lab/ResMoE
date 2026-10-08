"""Experiment 253 — OFFLINE CONVERSION COST of the deployable ResMoE pipeline (exp230: keep-pattern
grouping K=128 + x-router + rank-128 predicted correction). Wall-clock and peak memory of each one-time
stage, summed over all layers:
  calibration : dense forward over N_CALIB tokens, harvest FFN input x / activation a, mean statistics
  SVD (PCA)   : SVD of the centered FFN input -> 512-dim router/correction features      (shared by keeps)
  grouping    : keep-indicator + weighted k-means + balanced assignment                  (per keep)
  router      : per-group residual-norm targets + x-router MLP training                  (per keep)
  correction  : dropped-output error E under the x-router's selection (targets), SVD of E -> basis B_r,
                predictor MLP training                                                   (per keep)
  gating      : validation gating - layer by layer, run N_VAL held-out tokens through the converted prefix
                and keep a layer's correction only if its in-situ R2 = 1-||E-e_hat||^2/||E||^2 > 0 (per keep)
Peak memory per stage: GPU = torch max_memory_allocated (includes the resident fp16 model) and
max_memory_reserved; host = sampled process RSS. Model load / tokenization are reported but are not
conversion. A ppl check (dense / x-router / +corr on all layers / +corr gated) confirms the timed pipeline works.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/253_conversion_cost.py
Env: KEEPS (0.50,0.25), NCALIB (8192), STEPS router/correction MLP steps (3000), OUT=json path.
(results/253/*_run*.json are from the version without the gating stage; *_gated_run*.json include it.)
"""
from __future__ import annotations
import sys, pathlib, gc, os, json, time, threading
from collections import defaultdict
from contextlib import contextmanager
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.reproducibility import load_mbpp_code
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = int(os.environ.get("NCALIB", "8192"))
N_EVAL = 1024
N_VAL = 1024                                                # gating tokens, disjoint from calibration and eval
CHUNK = 512
K = 128
RCORR = 128
KEEPS = [float(k) for k in os.environ.get("KEEPS", "0.50,0.25").split(",")]
STEPS = int(os.environ.get("STEPS", "3000"))
OUT = os.environ.get("OUT", "")
STAGES = [('group', "grouping (k-means + balanced assign)"), ('router', "router training (targets + MLP)"),
          ('corr_tgt', "correction: error targets"), ('corr_svd', "SVD: correction basis B_r"),
          ('corr_fit', "correction training (MLP)"), ('gate', "validation gating (in-situ R2)")]


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


def rss_bytes():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class Stop(Exception):
    pass


class Meter:
    """accumulates wall-clock per stage (summed over calls) and peak GPU / host memory (max over calls)."""
    def __init__(self):
        self.t = defaultdict(float); self.alloc = defaultdict(int); self.resv = defaultdict(int)
        self.rss = defaultdict(int); self._peak = 0
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        while True:
            self._peak = max(self._peak, rss_bytes()); time.sleep(0.01)

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); self._peak = rss_bytes()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize(); self.t[name] += time.perf_counter() - t0
        self.alloc[name] = max(self.alloc[name], torch.cuda.max_memory_allocated())
        self.resv[name] = max(self.resv[name], torch.cuda.max_memory_reserved())
        self.rss[name] = max(self.rss[name], self._peak, rss_bytes())


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda"
    M = Meter(); GB = 1024 ** 3
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = load_mbpp_code()
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    torch.set_grad_enabled(False)
    # warm-up so lazy CUDA / cuSOLVER / autograd initialisation is not charged to the first stage
    model(ids[:CHUNK].unsqueeze(0).to(dev)); torch.linalg.svd(torch.randn(256, 64, device=dev), full_matrices=False)
    mlp_fit(torch.randn(64, 8, device=dev), torch.randn(64, 4, device=dev), dev, steps=3)
    torch.cuda.synchronize(); t_load = time.perf_counter() - t0
    model_gb = torch.cuda.memory_allocated() / GB
    print(f"  MODEL={MODEL} layers={nL} K={K} Rcorr={RCORR} N_CALIB={N_CALIB} STEPS={STEPS}", flush=True)
    print(f"  {torch.cuda.get_device_name(0)} | torch {torch.__version__} | load+warm-up {t_load:.1f}s | "
          f"resident model {model_gb:.2f} GB GPU, host RSS {rss_bytes()/GB:.2f} GB", flush=True)

    CFG = {li: {'active': False} for li in range(nL)}
    XC = {li: None for li in range(nL)}
    GATE = dict(li=None, num=0.0, den=0.0, Wd=None, E=None)  # layer whose correction is being validated

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
            sg = cfg['selr']((XC[li].float() - cfg['xbar']) @ cfg['P']).float()
            m = keep_topB_group(sg, cfg['gsz'], cfg['gf'], cfg['B']).to(a.dtype)
            if GATE['li'] == li:                            # true dropped-output error on the in-situ input
                GATE['E'] = ((af.float() - cfg['abar']) * (1 - m.float())) @ GATE['Wd'].T
            return ((af * m + cfg['abar'].to(a.dtype) * (1 - m)).reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            cfg = CFG[li]; gating = GATE['li'] == li
            if not cfg['active'] or not (cfg.get('corr') or gating):
                return None
            sh = output.shape; z = (XC[li].float() - cfg['xbar']) @ cfg['P']
            ehat = cfg['pred'](z) @ cfg['Br'].T
            if gating:
                GATE['num'] += float(((GATE['E'] - ehat) ** 2).sum()); GATE['den'] += float((GATE['E'] ** 2).sum())
                raise Stop
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    @torch.no_grad()
    def ppl():
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return float(torch.tensor(tot / ntok).exp())

    PPL = {'dense': ppl()}; R2 = {}; DROP = {}

    # ---- calibration: forward + harvest (x, a) per layer + mean statistics ----
    STR = {}
    with M.stage('calib'):
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
        dff = capa[0][0].shape[1]
        for li in range(nL):
            a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
            Wu = layers[li].mlp.up_proj.weight.detach().float().to(dev)
            Wd = downs[li].weight.detach().float().to(dev)
            af = a.float().to(dev)
            STR[li] = dict(a=a, x=x.half().cpu(), Wd=Wd.cpu(), abar=af.mean(0), ebar=(x @ Wu.T).abs().mean(0),
                           vn=Wd.norm(dim=0), xbar=x.mean(0))
            capa[li] = None; capx[li] = None
            del af, Wu, x; gc.collect(); torch.cuda.empty_cache()
    # ---- SVD (input PCA): router / correction features ----
    with M.stage('pca'):
        for li in range(nL):
            s = STR[li]; x = s['x'].float().to(dev)
            _, _, VtX = torch.linalg.svd(x - s['xbar'], full_matrices=False); s['P'] = VtX[:512].T
            del x, VtX; gc.collect(); torch.cuda.empty_cache()
    print(f"  dff={dff} hidden={STR[0]['xbar'].shape[0]}", flush=True)

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

    def train_selr(li, gf):
        """x-router: predict per-group residual norm from PCA(x)."""
        s = STR[li]; a = s['a'].float().to(dev); Wd = s['Wd'].to(dev); dev_a = a - s['abar']
        rn = torch.zeros(a.shape[0], K, device=dev)
        for g in range(K):
            ix = (gf == g).nonzero().flatten()
            if len(ix):
                rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        sr = mlp_fit(z, rn, dev, steps=STEPS)
        del a, Wd, dev_a, rn, z; gc.collect(); torch.cuda.empty_cache()
        return sr

    def train_corr(li, bf, gsz, gf, selr):
        s = STR[li]; B = int(round(bf * dff))
        with M.stage(f"{bf}|corr_tgt"):                     # dropped-output error under the x-router's selection
            a = s['a'].float().to(dev); Wd = s['Wd'].to(dev); x = s['x'].float().to(dev)
            z = (x - s['xbar']) @ s['P']
            m = keep_topB_group(selr(z).float(), gsz, gf, B).to(a.dtype)
            E = (a - (a * m + s['abar'] * (1 - m))) @ Wd.T
        with M.stage(f"{bf}|corr_svd"):
            _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
        with M.stage(f"{bf}|corr_fit"):
            pred = mlp_fit(z, E @ Br, dev, steps=STEPS)
            del a, Wd, x, m, E, Vt, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    for bf in KEEPS:
        GRP = {}; SELR = {}; BR = {}; PRED = {}
        for li in range(nL):
            with M.stage(f"{bf}|group"):
                GRP[li] = grouping(li, bf)
        for li in range(nL):
            with M.stage(f"{bf}|router"):
                SELR[li] = train_selr(li, GRP[li][1])
        for li in range(nL):
            BR[li], PRED[li] = train_corr(li, bf, *GRP[li], SELR[li])
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=True, B=int(round(bf * dff)), abar=s['abar'], gsz=gsz, gf=gf,
                                xbar=s['xbar'], P=s['P'], selr=SELR[li], corr=False, Br=BR[li], pred=PRED[li]))
        kept = []; R2[bf] = []
        with M.stage(f"{bf}|gate"):                         # greedy: layer li is judged with layers < li as decided
            for li in range(nL):
                GATE.update(li=li, num=0.0, den=0.0, Wd=STR[li]['Wd'].to(dev))
                for c0 in range(N_CALIB + N_EVAL, N_CALIB + N_EVAL + N_VAL, CHUNK):
                    try:
                        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
                    except Stop:
                        pass
                R2[bf].append(1 - GATE['num'] / GATE['den'])
                if R2[bf][-1] > 0:
                    kept.append(li); CFG[li]['corr'] = True
            GATE.update(li=None, Wd=None, E=None)
        DROP[bf] = [li for li in range(nL) if li not in kept]
        for tag, on in (('nocorr', []), ('corr_all', range(nL)), ('corr', kept)):   # sanity: the timed pipeline works
            for li in range(nL):
                CFG[li]['corr'] = li in on
            PPL[f"{bf}|{tag}"] = ppl()
        for li in range(nL):
            CFG[li]['active'] = False
        del GRP, SELR, BR, PRED; gc.collect(); torch.cuda.empty_cache()

    def row(name, key):
        print(f"  {name:<38} | {M.t[key]:>8.1f} | {M.t[key]/nL:>7.2f} | {M.alloc[key]/GB:>9.2f} | "
              f"{M.resv[key]/GB:>8.2f} | {M.rss[key]/GB:>8.2f}", flush=True)

    hdr = f"  {'stage':<38} | {'time (s)':>8} | {'s/layer':>7} | {'GPU alloc':>9} | {'GPU resv':>8} | {'host RSS':>8}"
    print(f"\n  peak memory in GB; GPU alloc includes the resident model ({model_gb:.2f} GB)\n{hdr}", flush=True)
    print("  -- shared across keep rates --", flush=True)
    row(f"calibration ({N_CALIB} tok fwd + stats)", 'calib'); row("SVD: input PCA (features)", 'pca')
    shared = M.t['calib'] + M.t['pca']; RES_K = {}
    for bf in KEEPS:
        print(f"  -- keep{int(bf*100)} --", flush=True)
        for key, name in STAGES:
            row(name, f"{bf}|{key}")
        per = sum(M.t[f"{bf}|{key}"] for key, _ in STAGES)
        recal = M.t['calib'] + sum(M.t[f"{bf}|{k}"] for k in ('corr_tgt', 'corr_svd', 'corr_fit', 'gate'))
        RES_K[f"{bf}"] = dict(total_s=shared + per, per_keep_s=per, recalibration_s=recal,
                              ppl_nocorr=PPL[f"{bf}|nocorr"], ppl_corr_all=PPL[f"{bf}|corr_all"],
                              ppl_corr=PPL[f"{bf}|corr"], gate_dropped=DROP[bf], gate_r2=R2[bf])
        print(f"  keep{int(bf*100)} TOTAL conversion (shared + per-keep) {shared + per:.1f}s = {(shared + per)/60:.1f} min"
              f" | correction-only recalibration (calibration + correction + gating) {recal:.1f}s", flush=True)
        print(f"  keep{int(bf*100)} ppl check: dense {PPL['dense']:.3f} | x-router {PPL[f'{bf}|nocorr']:.3f} -> +corr all "
              f"layers {PPL[f'{bf}|corr_all']:.3f} -> +corr gated {PPL[f'{bf}|corr']:.3f} (gating dropped layers {DROP[bf]})",
              flush=True)
    peak = dict(gpu_alloc_gb=max(M.alloc.values()) / GB, gpu_reserved_gb=max(M.resv.values()) / GB,
                host_rss_gb=max(M.rss.values()) / GB)
    print(f"\n  PEAK over all stages: GPU alloc {peak['gpu_alloc_gb']:.2f} GB | GPU reserved "
          f"{peak['gpu_reserved_gb']:.2f} GB | host RSS {peak['host_rss_gb']:.2f} GB", flush=True)
    if OUT:
        res = dict(model=MODEL, gpu=torch.cuda.get_device_name(0), torch=torch.__version__, layers=nL, dff=dff,
                   K=K, rcorr=RCORR, n_calib=N_CALIB, steps=STEPS, load_s=t_load, model_gpu_gb=model_gb,
                   stages={k: dict(time_s=M.t[k], gpu_alloc_gb=M.alloc[k] / GB, gpu_reserved_gb=M.resv[k] / GB,
                                   host_rss_gb=M.rss[k] / GB) for k in M.t},
                   keeps=RES_K, peak=peak, ppl=PPL)
        pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(OUT).write_text(json.dumps(res, indent=1))
        print(f"  wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
