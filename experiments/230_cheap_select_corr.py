"""Experiment 230 — does CHEAP selection + correction match gate+correction? exp229 showed the low-rank
correction absorbs routing error (gate-sel+corr ≈ oracle-sel+corr), so selection quality may be nearly
irrelevant. Test the CHEAPEST selection: x-router (learned MLP on PCA(x) -> group scores, ~free) + the
predicted correction. If (xrouter,+corr) ≈ (gate,+corr) ≈ (oracle,+corr) => cheapest full deployable =
x-router (~free select) + low-rank correction (~3% FFN), dropping the 33% gate cost. Qwen SwiGLU,
keep-pattern K128, corr rank128, keep50/25. Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/230_cheap_select_corr.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.reproducibility import configure_determinism, report_provenance, report_metrics
import torch, torch.nn as nn, torch.nn.functional as F

SEED = int(os.environ.get("SEED", "0"))
configure_determinism(SEED)

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
N_EVAL = 1024
CHUNK = 512
K = 128
RCORR = 128
KEEPS = [0.50, 0.25]
CUR_BF: float = 0.0


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


def mlp_fit(
    X: torch.Tensor, Y: torch.Tensor, dev: str, steps: int = 3000,
    hidden: int = 512, lr: float = 3e-3, bs: int = 2048,
) -> nn.Sequential:
    torch.manual_seed(SEED)
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
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, gf]


def sel_mask(sel, a, x, cfg, B):
    if sel == 'oracle':
        per = ((a - cfg['abar']) * cfg['vn']) ** 2
        sg = torch.zeros(a.shape[0], cfg['gsz'].shape[0], device=a.device).index_add_(1, cfg['gf'], per)
    elif sel == 'gate':
        h = F.silu(x @ cfg['Wg'].T); per = (h.abs() * cfg['ebar'] * cfg['vn']) ** 2
        sg = torch.zeros(a.shape[0], cfg['gsz'].shape[0], device=a.device).index_add_(1, cfg['gf'], per)
    else:  # xrouter
        z = (x - cfg['xbar']) @ cfg['P']
        sg = cfg['selr'](z).float()
    return keep_topB_group(sg, cfg['gsz'], cfg['gf'], B).to(a.dtype)


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code: list[str] = []
    fingerprints: dict[str, str] = {}
    for sp in ["train", "test", "validation", "prompt"]:
        dataset = load_dataset("google-research-datasets/mbpp", "full", split=sp)
        fingerprints[sp] = dataset._fingerprint
        code.extend(dataset["code"])
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    print(f"  MODEL={MODEL} [SwiGLU] layers={nL} K={K} Rcorr={RCORR}", flush=True)
    torch.set_grad_enabled(False)
    report_provenance(source=__file__, model=MODEL, revision=model.config._commit_hash,
                      seed=SEED, tokens=ids, fingerprints=fingerprints,
                      calibration_items=N_CALIB, evaluation_items=N_EVAL, chunk=CHUNK)

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
            a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1]); x = XC[li].float()
            B = cfg['B']
            if cfg.get('neuron'):
                m = keep_topB_neuron(((af.float() - cfg['abar']).abs() * cfg['vn']), B).to(a.dtype)
            else:
                m = sel_mask(cfg['sel'], af.float(), x, cfg, B).to(a.dtype)
            masked = af * m + cfg['abar'].to(a.dtype) * (1 - m)
            return (masked.reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            cfg = CFG[li]
            if not cfg['active'] or not cfg.get('corr'):
                return None
            x = XC[li].float(); sh = output.shape
            z = (x - cfg['xbar']) @ cfg['P']
            ehat = cfg['pred'](z)[:, :cfg['r']] @ cfg['Br'][:, :cfg['r']].T
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    @torch.no_grad()
    def ce_eval():
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return tot / ntok

    def ppl():
        return float(torch.tensor(ce_eval()).exp())

    dense_ppl = ppl()
    report_metrics(dict(seed=SEED, configuration="dense", ppl=dense_ppl))
    print(f"  dense ppl {dense_ppl:.3f}", flush=True)

    capa: dict[int, list[torch.Tensor]] = {li: [] for li in range(nL)}; capx: dict[int, list[torch.Tensor]] = {li: [] for li in range(nL)}
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
    print(f"  dff={dff}\n", flush=True)

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wg = layers[li].mlp.gate_proj.weight.detach().float().to(dev)
        Wu = layers[li].mlp.up_proj.weight.detach().float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0); ebar = (x @ Wu.T).abs().mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), Wg=Wg.cpu(), Wd=Wd.cpu(), abar=abar, ebar=ebar,
                       vn=vn, xbar=xbar, P=VtX[:512].T)
        del capa[li], capx[li]
        del af, Wu; gc.collect(); torch.cuda.empty_cache()

    def grouping(li: int, bf: float) -> tuple[torch.Tensor, torch.Tensor]:
        s = STR[li]; a = s['a'].float().to(dev); vn = s['vn']; abar = s['abar']; B = int(round(bf * dff))
        Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
        w = ((a - abar).abs() * vn).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=SEED))
        gsz = torch.zeros(K, device=dev)
        for g in range(K):
            gsz[g] = (gl == g).sum()
        del a, Bind; gc.collect(); torch.cuda.empty_cache()
        return gsz, gl

    def train_selr(li, bf, gsz, gf):
        """x-router: predict per-group residual norm from PCA(x)."""
        s = STR[li]; a = s['a'].float().to(dev); Wd = s['Wd'].to(dev); dev_a = a - s['abar']
        rn = torch.zeros(a.shape[0], K, device=dev)
        for g in range(K):
            ix = (gf == g).nonzero().flatten()
            if len(ix):
                rn[:, g] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        sr = mlp_fit(z, rn, dev)
        del a, Wd, dev_a, rn, z; gc.collect(); torch.cuda.empty_cache()
        return sr

    def train_corr(li, bf, gsz, gf, sel, SELR):
        s = STR[li]; a = s['a'].float().to(dev); Wd = s['Wd'].to(dev); x = s['x'].float().to(dev)
        B = int(round(bf * dff))
        cfg = dict(abar=s['abar'], vn=s['vn'], gsz=gsz, gf=gf, Wg=s['Wg'].to(dev), ebar=s['ebar'],
                   xbar=s['xbar'], P=s['P'], selr=SELR[li] if SELR else None)
        m = sel_mask(sel, a, x, cfg, B)
        drop = a - (a * m + s['abar'] * (1 - m))
        E = drop @ Wd.T
        _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
        c = E @ Br; z = (x - s['xbar']) @ s['P']
        pred = mlp_fit(z, c, dev)
        del a, Wd, x, m, drop, E, Vt, c, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def setrun(GRP, sel, SELR=None, corr=False, BR=None, PRED=None, neuron=False):
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=True, B=int(round(CUR_BF * dff)), abar=s['abar'], vn=s['vn'],
                                gsz=gsz, gf=gf, Wg=s['Wg'].to(dev), ebar=s['ebar'], xbar=s['xbar'], P=s['P'],
                                sel=sel, selr=SELR[li] if SELR else None, corr=corr, r=RCORR,
                                Br=BR[li] if BR else None, pred=PRED[li] if PRED else None, neuron=neuron))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    print(f"  SwiGLU ppl (dense {dense_ppl:.3f}), keep-pattern K={K}, corr rank={RCORR}\n", flush=True)
    global CUR_BF
    for bf in KEEPS:
        CUR_BF = bf
        GRP = {li: grouping(li, bf) for li in range(nL)}
        SELR = {li: train_selr(li, bf, *GRP[li]) for li in range(nL)}
        setrun(GRP, 'oracle', neuron=True); nu = ppl(); off()
        report_metrics(dict(seed=SEED, keep=bf, dense=dense_ppl, neuron_oracle=nu))
        print(f"  keep{int(bf*100)}: dense {dense_ppl:.3f} | neu-oracle {nu:.3f}", flush=True)
        for sel in ['oracle', 'gate', 'xrouter']:
            setrun(GRP, sel, SELR=SELR); nc = ppl(); off()
            BR = {}; PRED = {}
            for li in range(nL):
                BR[li], PRED[li] = train_corr(li, bf, *GRP[li], sel, SELR)
            setrun(GRP, sel, SELR=SELR, corr=True, BR=BR, PRED=PRED); cc = ppl(); off()
            report_metrics(dict(seed=SEED, keep=bf, selector=sel, uncorrected=nc, corrected=cc))
            tag = "  <-- CHEAPEST DEPLOYABLE" if sel == 'xrouter' else ""
            print(f"           ({sel:>7} sel) no-corr {nc:.3f} -> +corr {cc:.3f}{tag}", flush=True)
        print("", flush=True)
    print("READ: if (xrouter,+corr) ≈ (gate,+corr) ≈ (oracle,+corr) => selection is irrelevant given the", flush=True)
    print("correction => cheapest full deployable = x-router (~free) + low-rank correction (~3% FFN).", flush=True)


if __name__ == "__main__":
    main()
