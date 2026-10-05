"""Experiment 252 — DIRECT routing metrics for the RQ1 selector table (exp224/225). Perplexity alone
conflates selector quality with downstream error propagation: the group "oracle" ranks groups by LOCAL
contribution, it does not minimize ppl (gate beats it at keep25). Here the selectors are compared
directly, same setup as exp224 (Qwen SwiGLU, keep-pattern grouping K=128, grp+mean representative):
  (1) group-selection OVERLAP per token/layer: Jaccard |A∩B|/|A∪B|, recall |A∩B|/|B| of the oracle
      set, share of the oracle's contribution mass captured, recall of the neuron-oracle keep set
  (2) FFN-output RECONSTRUCTION error ||y_hat - y||^2 / ||y||^2, y = W_d·a (dense FFN output)
Two regimes:
  teacher-forced : dense forward, every selector sees the SAME dense FFN input (isolates the selector)
  in-situ        : selector active in all layers (the ppl setting); overlap + local error on its own
                   drifted input, plus accumulated FFN-output error vs the dense run's output
Selectors: oracle | gate | x-router. References: rn-oracle (the x-router's regression target computed
from true activations = what a perfect x-router would select), static (groups most often oracle-kept
on calibration; input-independent), random (chance), neu-oracle (per-neuron ceiling), none (all dropped).
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/252_routing_direct_metrics.py
Env: GREF=keep regroup per keep (exp225, default) | GREF=0.5 fixed keep50 grouping (exp224);
     NEVAL eval tokens (1024), STEPS x-router steps (3000, as exp230/254), SEED router init/batches, OUT=json path.
     Grouping, statistics and correction targets use the fp16 activations captured in the calibration forward
     (as exp230/253/254); runs before 2026-10-04 recomputed them in fp32 from x and used 2500 router steps.
     CORR=1 also trains a rank-128 correction on each selector's own selection error (exp230) and reports
     the FFN-output error AFTER correction ('<sel>+corr' rows: teacher-forced, in-situ, and ppl).
"""
from __future__ import annotations
import sys, pathlib, gc, os, json
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = int(os.environ.get("NEVAL", "1024"))
CHUNK = 512
K = 128
KEEPS = [float(k) for k in os.environ.get("KEEPS", "0.50,0.25").split(",")]
GREF = os.environ.get("GREF", "keep")
STEPS = int(os.environ.get("STEPS", "3000"))
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", "")
CORR = os.environ.get("CORR", "0") == "1"
RCORR = 128
MAIN = ['oracle', 'gate', 'xrouter']
SEL = MAIN + ['rn-oracle', 'static', 'random']              # group selectors
ALL = SEL + ['neu-oracle']


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


def mlp_fit(X, Y, dev, steps=2500, lr=3e-3, bs=2048, hidden=512):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(dev).float()
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


def select_groups(score_g, gsz, B):
    """top-scoring groups until B neurons are covered -> [n,K] bool group mask (exp224 keep_topB_group)."""
    order = score_g.argsort(1, descending=True); so = gsz[order]
    return torch.zeros_like(score_g, dtype=torch.bool).scatter_(1, order, (so.cumsum(1) - so) < B)


def group_sum(per, gf, Kc):
    return torch.zeros(per.shape[0], Kc, device=per.device).index_add_(1, gf, per)


def group_resnorm(dev_a, Wd, gix):
    """per-group dropped-output norm ||sum_{k in g}(a_k - abar_k) v_k|| (the x-router target)."""
    rn = torch.zeros(dev_a.shape[0], len(gix), device=dev_a.device)
    for g, ix in enumerate(gix):
        if len(ix):
            rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
    return rn


def group_scores(mode, s, g, bf, dev_a, x, rgen):
    """dev_a:[n,dff] TRUE activation minus calib mean. x:[n,hidden] FFN input. -> [n,K] group scores."""
    if mode == 'oracle':
        return group_sum((dev_a * s['vn']) ** 2, g['gf'], K)
    if mode == 'gate':                                      # |gate| * mean|up| * ||v|| proxy (1/3 FFN)
        return group_sum((F.silu(x @ s['Wg'].T).abs() * s['ebar'] * s['vn']) ** 2, g['gf'], K)
    if mode == 'xrouter':
        return g['router']((x - s['xbar']) @ s['P']).float()
    if mode == 'rn-oracle':
        return group_resnorm(dev_a, s['Wd'], g['gix'])
    if mode == 'static':
        return g['freq'][bf].expand(dev_a.shape[0], K)
    return torch.rand(dev_a.shape[0], K, device=dev_a.device, generator=rgen)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = []
    for sp in ["train", "test", "validation", "prompt"]:
        try:
            code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    assert len(ids) >= N_CALIB + N_EVAL and N_EVAL % CHUNK == 0, (len(ids), N_CALIB, N_EVAL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gates = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    print(f"  MODEL={MODEL} [SwiGLU] layers={nL} K={K} GREF={GREF} NEVAL={N_EVAL} STEPS={STEPS} SEED={SEED}", flush=True)
    torch.set_grad_enabled(False)
    rgen = torch.Generator(device=dev).manual_seed(SEED)

    STR = {}                                                # per-layer calibration statistics
    GRP = {}                                                # keep -> layer -> grouping/router
    RUN = {'mode': None, 'bf': None, 'tf': False, 'c0': 0, 'corr': False}
    XC = {li: None for li in range(nL)}
    CORRS = {}                                              # (keep, selector) -> layer -> (Br, pred)
    EHAT = {}
    DENSE_Y = {}                                            # (layer, chunk) -> dense FFN output
    ACC = defaultdict(float)                                # (regime, keep, selector, layer, stat) -> sum
    TOK = defaultdict(lambda: torch.zeros(N_EVAL, device=dev))  # per-token, summed over layers

    def add(regime, bf, m, li, **stats):
        for k, v in stats.items():
            ACC[regime, bf, m, li, k] += v.sum().double()

    def recon(dev_a, mk, Wd):
        """dropped-output error of keep mask mk with the mean representative: y - y_hat."""
        return (dev_a * ~mk) @ Wd.T

    def teacher_forced(li, a, x):
        s = STR[li]; Wd = s['Wd']; n = a.shape[0]; t0 = RUN['c0'] - N_CALIB
        dev_a = a - s['abar']; y = a @ Wd.T; y2 = (y * y).sum(1).clamp(min=1e-12)
        DENSE_Y[li, RUN['c0']] = y
        con2 = (dev_a * s['vn']) ** 2
        for bf in KEEPS:
            g = GRP[bf][li]; B = int(round(bf * a.shape[1]))
            sc = {m: group_scores(m, s, g, bf, dev_a, x, rgen) for m in SEL}
            sel = {m: select_groups(sc[m], g['gsz'], B) for m in SEL}
            neu = keep_topB_neuron(con2, B)
            masks = {m: sel[m][:, g['gf']] for m in SEL}
            masks['neu-oracle'] = neu; masks['none'] = torch.zeros_like(neu)
            mass_or = (sc['oracle'] * sel['oracle']).sum(1).clamp(min=1e-12)
            for m, mk in masks.items():
                e = recon(dev_a, mk, Wd); err2 = (e * e).sum(1)
                add('tf', bf, m, li, err2=err2, y2=y2, rel=err2 / y2,
                    cos=F.cosine_similarity(y - e, y, dim=1), nrec=(mk & neu).sum(1) / B)
                TOK['rel', bf, m][t0:t0 + n] += err2 / y2
                if m in sel:
                    add('tf', bf, m, li, mass=(sc['oracle'] * sel[m]).sum(1) / mass_or)
                if (bf, m) in CORRS:                        # same selection, error left after the correction
                    Br, pred = CORRS[bf, m][li]; ec = e - pred((x - s['xbar']) @ s['P']) @ Br.T; ec2 = (ec * ec).sum(1)
                    add('tf', bf, m + '+corr', li, err2=ec2, y2=y2, rel=ec2 / y2, cos=F.cosine_similarity(y - ec, y, dim=1))
                    TOK['rel', bf, m + '+corr'][t0:t0 + n] += ec2 / y2
            for m1 in SEL:
                for m2 in SEL:
                    inter = (sel[m1] & sel[m2]).sum(1).float()
                    jac = inter / (sel[m1] | sel[m2]).sum(1)
                    add('tf', bf, f"{m1}|{m2}", li, jac=jac, rec=inter / sel[m2].sum(1))
                    if m2 == 'oracle':
                        TOK['jac', bf, m1][t0:t0 + n] += jac

    def routed(li, a16, x):
        """in-situ: mask the TRUE activation with RUN['mode'] (fill dropped with mean) + log direct metrics."""
        s = STR[li]; Wd = s['Wd']; bf = RUN['bf']; m = RUN['mode']; g = GRP[bf][li]
        a = a16.float(); B = int(round(bf * a.shape[1]))
        dev_a = a - s['abar']; con2 = (dev_a * s['vn']) ** 2
        if m == 'neu-oracle':
            mk = keep_topB_neuron(con2, B)
        else:
            sel_or = select_groups(group_sum(con2, g['gf'], K), g['gsz'], B)
            selm = sel_or if m == 'oracle' else select_groups(group_scores(m, s, g, bf, dev_a, x, rgen), g['gsz'], B)
            mk = selm[:, g['gf']]
            inter = (selm & sel_or).sum(1).float()
            add('is', bf, m, li, jac=inter / (selm | sel_or).sum(1), rec=inter / sel_or.sum(1))
        y = a @ Wd.T; y2 = (y * y).sum(1).clamp(min=1e-12)   # dense FFN output on the drifted input
        e = recon(dev_a, mk, Wd)
        if RUN['corr']:                                     # correction trained on this selector's own error
            Br, pred = CORRS[bf, m][li]; EHAT[li] = pred((x - s['xbar']) @ s['P']) @ Br.T; e = e - EHAT[li]
        err2 = (e * e).sum(1)
        yd = DENSE_Y[li, RUN['c0']]; acc2 = ((y - e - yd) ** 2).sum(1); yd2 = (yd * yd).sum(1).clamp(min=1e-12)
        add('is', bf, m + ('+corr' if RUN['corr'] else ''), li, err2=err2, y2=y2, rel=err2 / y2, acc2=acc2, yd2=yd2, accrel=acc2 / yd2)
        return torch.where(mk, a16, s['abar'].to(a16.dtype))

    def gpre(li):
        def hook(_m, args):
            XC[li] = args[0].reshape(-1, args[0].shape[-1])
        return hook

    def dpre(li):
        def hook(_m, args):
            a = args[0]; sh = a.shape
            if RUN['tf']:
                teacher_forced(li, a.reshape(-1, sh[-1]).float(), XC[li].float())
                return None
            if RUN['mode'] is None:
                return None
            return (routed(li, a.reshape(-1, sh[-1]), XC[li].float()).reshape(sh),) + args[1:]
        return hook
    def dpost(li):
        def hook(_m, args, output):
            if RUN['mode'] is None or not RUN['corr']:
                return None
            sh = output.shape
            return (output.reshape(-1, sh[-1]) + EHAT[li].to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gates[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    @torch.no_grad()
    def ppl(mode=None, bf=None, tf=False, corr=False):
        RUN.update(mode=mode, bf=bf, tf=tf, corr=corr)
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            RUN['c0'] = c0
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        RUN.update(mode=None, bf=None, tf=False, corr=False)
        return float(torch.tensor(tot / ntok).exp())

    # harvest FFN input x (gate input) and activation a (down_proj input) per layer on the calibration tokens
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = [gates[li].register_forward_pre_hook(
        (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li))
        for li in range(nL)]
    hs += [downs[li].register_forward_pre_hook(
        (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li))
        for li in range(nL)]
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()

    dff = None
    for li in range(nL):
        x = torch.cat(capx[li]).float().to(dev); capx[li] = None
        a = torch.cat(capa[li]); capa[li] = None                 # fp16 activations as seen by the model
        Wg = layers[li].mlp.gate_proj.weight.detach().float()
        Wu = layers[li].mlp.up_proj.weight.detach().float()
        Wd = layers[li].mlp.down_proj.weight.detach().float()
        dff = Wg.shape[0]
        u = x @ Wu.T
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(x=x.half().cpu(), a=a, Wg=Wg, Wu=Wu, Wd=Wd, vn=Wd.norm(dim=0), abar=a.float().to(dev).mean(0),
                       ebar=u.abs().mean(0), xbar=xbar, P=Vt[:512].T)
        del x, u, Vt; gc.collect(); torch.cuda.empty_cache()
    print(f"  hidden={STR[0]['xbar'].shape[0]} dff={dff}", flush=True)

    def build(li, bref):
        """keep-pattern grouping at reference keep bref + x-router + static oracle-keep frequencies."""
        s = STR[li]; x = s['x'].float().to(dev)
        a = s['a'].float().to(dev); dev_a = a - s['abar']; con = dev_a.abs() * s['vn']
        Bind = keep_topB_neuron(con, int(round(bref * dff))).float()
        gf = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), con.mean(0), K, seed=0))
        gsz = torch.bincount(gf, minlength=K).float()
        gix = [(gf == g).nonzero().flatten() for g in range(K)]
        router = mlp_fit((x - s['xbar']) @ s['P'], group_resnorm(dev_a, s['Wd'], gix), dev, steps=STEPS)
        sg = group_sum(con ** 2, gf, K)
        freq = {bf: select_groups(sg, gsz, int(round(bf * dff))).float().mean(0) for bf in KEEPS}
        del x, a, dev_a, con, Bind, sg; gc.collect(); torch.cuda.empty_cache()
        return dict(gf=gf, gsz=gsz, gix=gix, router=router, freq=freq)

    built = {}
    for bf in KEEPS:
        bref = bf if GREF == "keep" else float(GREF)
        if bref not in built:
            built[bref] = {li: build(li, bref) for li in range(nL)}
            print(f"  grouping=keep-pattern(ref keep{int(bref*100)}), x-router trained", flush=True)
        GRP[bf] = built[bref]

    def train_corr(li, bf, m):
        """rank-RCORR basis + predictor of the dropped-output error under selector m (exp230)."""
        s = STR[li]; g = GRP[bf][li]; x = s['x'].float().to(dev); B = int(round(bf * dff))
        a = s['a'].float().to(dev); dev_a = a - s['abar']
        mk = select_groups(group_scores(m, s, g, bf, dev_a, x, rgen), g['gsz'], B)[:, g['gf']]
        E = recon(dev_a, mk, s['Wd']); _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
        z = (x - s['xbar']) @ s['P']; pred = mlp_fit(z, E @ Br, dev, steps=3000)
        del x, a, dev_a, mk, E, Vt, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    if CORR:
        for bf in KEEPS:
            for m in MAIN:
                CORRS[bf, m] = {li: train_corr(li, bf, m) for li in range(nL)}
        print(f"  rank-{RCORR} corrections trained for {MAIN} at keeps {KEEPS}", flush=True)
    PPL = {'dense': ppl(tf=True)}                           # dense pass also logs teacher-forced metrics
    for bf in KEEPS:
        for m in ALL:
            PPL[f"{bf}|{m}"] = ppl(m, bf)
        for m in (MAIN if CORR else []):
            PPL[f"{bf}|{m}+corr"] = ppl(m, bf, corr=True)

    def layer_stats(regime, bf, m):
        """per-layer means -> dict stat -> list over layers (nmse/accnmse = ratio of sums)."""
        out = defaultdict(list)
        for li in range(nL):
            st = {k[4]: float(v) for k, v in ACC.items() if k[:4] == (regime, bf, m, li)}
            if not st:
                return {}
            if 'err2' in st:
                out['nmse'].append(st.pop('err2') / st.pop('y2'))
            if 'acc2' in st:
                out['accnmse'].append(st.pop('acc2') / st.pop('yd2'))
            for k, v in st.items():
                out[k].append(v / N_EVAL)                   # every pass sees each eval token once per layer
        return dict(out)

    def avg(v):
        return sum(v) / len(v) if v else float('nan')

    def ci(key):                                            # 95% CI half-width over eval tokens
        t = TOK[key] / nL
        return 1.96 * float(t.std()) / N_EVAL ** 0.5

    RES = dict(model=MODEL, K=K, gref=GREF, n_calib=N_CALIB, n_eval=N_EVAL, steps=STEPS, seed=SEED,
               dff=dff, layers=nL, ppl=PPL, keeps={})
    print(f"\n  dense ppl {PPL['dense']:.3f}; keep-pattern K={K}, grp+mean; means over {nL} layers x {N_EVAL} "
          f"eval tokens (±95% CI over tokens)", flush=True)
    for bf in KEEPS:
        R = RES['keeps'][f"{bf}"] = dict(tf={}, insitu={}, pair_jaccard={}, pair_recall={})
        ng = int(select_groups(GRP[bf][0]['freq'][bf][None], GRP[bf][0]['gsz'], int(round(bf * dff))).sum())
        print(f"\n  ===== keep{int(bf*100)} ({ng}/{K} groups kept per token) =====", flush=True)
        print("  [teacher-forced: all selectors on the same dense input; Jaccard/recall/mass vs group-oracle set]", flush=True)
        print(f"  {'selector':>10} | {'ppl':>8} | {'Jaccard':>14} | {'recall':>6} | {'mass':>6} | "
              f"{'neuRecall':>9} | {'NMSE':>6} | {'relErr':>14} | {'cos':>6}", flush=True)
        for m in ALL + ['none'] + ([x + '+corr' for x in MAIN] if CORR else []):
            st = layer_stats('tf', bf, m); pr = layer_stats('tf', bf, f"{m}|oracle")
            R['tf'][m] = dict(per_layer=dict(st, **pr), ppl=PPL.get(f"{bf}|{m}"),
                              **{k: avg(v) for k, v in dict(st, **pr).items()})
            r = R['tf'][m]
            if pr:
                r['jac_ci95'] = ci(('jac', bf, m))
            r['rel_ci95'] = ci(('rel', bf, m))
            p = f"{r['ppl']:>8.3f}" if r['ppl'] is not None else f"{'-':>8}"
            jr = (f"{r['jac']:>6.3f} ±{r['jac_ci95']:.3f} | {r['rec']:>6.3f} | {r['mass']:>6.3f}" if pr
                  else f"{'-':>14} | {'-':>6} | {'-':>6}")
            print(f"  {m:>12} | {p} | {jr} | {r.get('nrec', float('nan')):>9.3f} | {r['nmse']:>6.3f} | "
                  f"{r['rel']:>6.3f} ±{r['rel_ci95']:.3f} | {r['cos']:>6.3f}", flush=True)
        print("  pairwise group-selection Jaccard (teacher-forced):", flush=True)
        print("  " + " " * 10 + " | " + " | ".join(f"{m:>9}" for m in SEL), flush=True)
        for m1 in SEL:
            row = []
            for m2 in SEL:
                pr = layer_stats('tf', bf, f"{m1}|{m2}")
                R['pair_jaccard'][f"{m1}|{m2}"] = avg(pr['jac']); R['pair_recall'][f"{m1}|{m2}"] = avg(pr['rec'])
                row.append(f"{avg(pr['jac']):>9.3f}")
            print(f"  {m1:>10} | " + " | ".join(row), flush=True)
        print("  [in-situ: selector active in all layers; overlap vs oracle + local error on its own drifted input;"
              " accNMSE = FFN-output error vs the DENSE run]", flush=True)
        print(f"  {'selector':>10} | {'ppl':>8} | {'Jaccard':>7} | {'recall':>6} | {'NMSE':>6} | {'relErr':>6} | "
              f"{'accNMSE':>7} | {'accRel':>6}", flush=True)
        for m in ALL + ([x + '+corr' for x in MAIN] if CORR else []):
            st = layer_stats('is', bf, m)
            R['insitu'][m] = dict(per_layer=st, ppl=PPL[f"{bf}|{m}"], **{k: avg(v) for k, v in st.items()})
            r = R['insitu'][m]
            jr = f"{r['jac']:>7.3f} | {r['rec']:>6.3f}" if 'jac' in r else f"{'-':>7} | {'-':>6}"
            print(f"  {m:>12} | {r['ppl']:>8.3f} | {jr} | {r['nmse']:>6.3f} | {r['rel']:>6.3f} | "
                  f"{r['accnmse']:>7.3f} | {r['accrel']:>6.3f}", flush=True)
        print("  per-layer (teacher-forced): Jaccard vs oracle [gate, x-router] | NMSE [oracle, gate, x-router]", flush=True)
        for li in range(nL):
            pl = {m: R['tf'][m]['per_layer'] for m in MAIN}
            print(f"    L{li:<2} | {pl['gate']['jac'][li]:.3f} {pl['xrouter']['jac'][li]:.3f} | "
                  f"{pl['oracle']['nmse'][li]:.3f} {pl['gate']['nmse'][li]:.3f} {pl['xrouter']['nmse'][li]:.3f}", flush=True)
    if OUT:
        pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(OUT).write_text(json.dumps(RES, indent=1))
        print(f"\n  wrote {OUT}", flush=True)
    print("\nREAD: Jaccard/recall = how often a selector picks the oracle's groups (random = chance, static ="
          " input-independent floor, rn-oracle = perfect x-router). NMSE (teacher-forced) = selector quality"
          " in FFN-output space, free of propagation; accNMSE (in-situ) = what the ppl column actually sees.", flush=True)


if __name__ == "__main__":
    main()
