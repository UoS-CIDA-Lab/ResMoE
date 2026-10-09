"""Experiment 228 — DEPLOYABLE low-rank correction of the grouping floor (A). exp227 (oracle ceiling):
a rank-r correction of the dropped-output error closes ~48%(keep50)/83%(keep25) of A. But that used the
TRUE error e(t). Here: PREDICT the correction coords c(t)=B_rᵀe(t) ∈ R^r from the FFN input x(t) (a cheap
always-on rank-r path), add ê=B_r·ĉ to the group-oracle output. Measure how much of the oracle-correction
ceiling SURVIVES prediction. Compare per r: group-oracle (no corr) | +oracle-corr (true e) | +predicted-corr
(from x). Qwen(SwiGLU), keep-pattern grouping K=128, oracle group select, keep50/25.
Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/228_deployable_correction.py
For limited GPU memory: CALIBRATION_DEVICE=cpu keeps all calibration tokens,
uses CPU SVD and grouping-weight averages, and batches distance/error calculations.
The default model-device setting preserves the original SVD layout and error GEMM size.
"""
from __future__ import annotations
import sys, pathlib, gc, os
from enum import Enum
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.reproducibility import configure_determinism, report_provenance, report_metrics
import torch, torch.nn as nn, torch.nn.functional as F

SEED = int(os.environ.get("SEED", "0"))
configure_determinism(SEED)

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = int(os.environ.get("NCALIB", "8192"))   # reduce for big models (7B) to fit GPU
N_EVAL = 1024
CHUNK = int(os.environ.get("CHUNK", "512"))       # reduce eval-forward memory for big models
K = 128
RMAX = 128
RANKS = [16, 32, 64, 128]
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.50,0.25").split(",")]
if not KEEPS or any(not 0 < keep <= 1 for keep in KEEPS):
    raise ValueError("KEEPS must contain fractions in (0, 1]")


class CalibrationDevice(Enum):
    MODEL = "model"
    CPU = "cpu"


CALIBRATION_DEVICE = CalibrationDevice(os.environ.get("CALIBRATION_DEVICE", "model"))


def centroid_distances(patterns: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Bound the distance calculation's temporary storage without dropping neurons."""
    batch_size = 4096 if CALIBRATION_DEVICE is CalibrationDevice.CPU else patterns.shape[0]
    return torch.cat([
        torch.cdist(patterns[c0:c0 + batch_size], centroids)
        for c0 in range(0, patterns.shape[0], batch_size)
    ])


def right_singular_basis(
    matrix: torch.Tensor, rank: int, center: torch.Tensor | None = None
) -> torch.Tensor:
    """Preserve the model-device SVD layout; compact the basis for CPU calibration."""
    if not 1 <= rank <= min(matrix.shape):
        raise ValueError("SVD rank must be within the matrix dimensions")
    svd_input = matrix.cpu() if CALIBRATION_DEVICE is CalibrationDevice.CPU else matrix
    if center is not None:
        svd_input = svd_input - center.to(svd_input.device)
    _, _, vectors = torch.linalg.svd(svd_input, full_matrices=False)
    if CALIBRATION_DEVICE is CalibrationDevice.MODEL:
        return vectors[:rank].T
    return vectors[:rank].T.clone(memory_format=torch.contiguous_format).to(matrix.device)


def weighted_kmeans_centroids(
    X: torch.Tensor, w: torch.Tensor, k: int, iters: int = 20, seed: int = 0
) -> torch.Tensor:
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = centroid_distances(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                wj = w[m]; c[j] = (X[m] * wj.unsqueeze(1)).sum(0) / wj.sum().clamp(min=1e-6)
    return c


def balanced_assign(X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    D = centroid_distances(X, centroids); n, k = D.shape
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
    torch.manual_seed(SEED)  # matched predictor initialization within each conversion seed
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


def oracle_mask(a, abar, vn, gsz, gf, B):
    per = ((a - abar) * vn) ** 2
    sg = torch.zeros(a.shape[0], gsz.shape[0], device=a.device).index_add_(1, gf, per)
    return keep_topB_group(sg, gsz, gf, B).to(a.dtype)


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
    print(f"  [mbpp load] ids tokens={ids.numel()}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    print(f"  MODEL={MODEL} [SwiGLU] layers={nL} K={K} ranks={RANKS} CALIBRATION_DEVICE={CALIBRATION_DEVICE.value}", flush=True)
    torch.set_grad_enabled(False)
    report_provenance(source=__file__, model=MODEL, revision=model.config._commit_hash,
                      seed=SEED, tokens=ids, fingerprints=fingerprints,
                      calibration_items=N_CALIB, evaluation_items=N_EVAL, chunk=CHUNK)

    CFG = {li: {'active': False} for li in range(nL)}
    XC = {li: None for li in range(nL)}; STASH = {li: None for li in range(nL)}

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
            B = cfg['B']
            if cfg.get('neuron'):
                m = keep_topB_neuron(((af.float() - cfg['abar']).abs() * cfg['vn']), B).to(a.dtype)
            else:
                m = oracle_mask(af.float(), cfg['abar'], cfg['vn'], cfg['gsz'], cfg['gf'], B).to(a.dtype)
            masked = af * m + cfg['abar'].to(a.dtype) * (1 - m)
            if cfg.get('corr') == 'oracle':
                STASH[li] = (af - masked).float()
            return (masked.reshape(sh),) + args[1:]
        return hook

    def dpost(li):
        def hook(_m, args, output):
            cfg = CFG[li]
            if not cfg['active'] or not cfg.get('corr'):
                return None
            r = cfg['r']; Br = cfg['Br'][:, :r]; sh = output.shape
            if cfg['corr'] == 'oracle':
                e = STASH[li] @ cfg['Wd'].to(STASH[li].device).T; STASH[li] = None
                ehat = (e @ Br) @ Br.T
            else:                                            # predicted from x
                z = (XC[li].float() - cfg['xbar']) @ cfg['P']
                chat = cfg['pred'](z)[:, :r]
                ehat = chat @ Br.T
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
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); P = right_singular_basis(x, 512, center=xbar)
        STR[li] = dict(a=a, x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=P)
        del capa[li], capx[li]
        del af, x, Wd; gc.collect(); torch.cuda.empty_cache()

    def grouping(li: int, bf: float) -> tuple[torch.Tensor, torch.Tensor]:
        s = STR[li]; vn = s['vn']; abar = s['abar']; B = int(round(bf * dff))
        bind_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        for c0 in range(0, s['a'].shape[0], CHUNK):
            a = s['a'][c0:c0 + CHUNK].float().to(dev)
            score = (a - abar).abs() * vn
            bind_chunks.append(keep_topB_neuron(score, B).float().cpu())
            score_chunks.append(score.cpu())
        patterns = torch.cat(bind_chunks).T.contiguous().to(dev)
        mean_device = torch.device("cpu") if CALIBRATION_DEVICE is CalibrationDevice.CPU else torch.device(dev)
        w = torch.cat(score_chunks).to(mean_device).mean(0).to(dev)
        gl = balanced_assign(patterns, weighted_kmeans_centroids(patterns, w, K, seed=SEED))
        gsz = torch.zeros(K, device=dev)
        for g in range(K):
            gsz[g] = (gl == g).sum()
        del a, score, patterns, bind_chunks, score_chunks; gc.collect(); torch.cuda.empty_cache()
        return gsz, gl

    def basis_and_pred(
        li: int, bf: float, gsz: torch.Tensor, gf: torch.Tensor
    ) -> tuple[torch.Tensor, nn.Sequential]:
        s = STR[li]; abar = s['abar']; vn = s['vn']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        error_chunks: list[torch.Tensor] = []
        batch_size = CHUNK if CALIBRATION_DEVICE is CalibrationDevice.CPU else s['a'].shape[0]
        for c0 in range(0, s['a'].shape[0], batch_size):
            a = s['a'][c0:c0 + batch_size].float().to(dev)
            m = oracle_mask(a, abar, vn, gsz, gf, B)
            drop = a - (a * m + abar * (1 - m))
            error_chunks.append((drop @ Wd.T).cpu())
        E = torch.cat(error_chunks).to(dev)                   # [N,d], all calibration tokens
        Br = right_singular_basis(E, RMAX)                    # [d,RMAX]
        c = E @ Br                                           # [N,RMAX] correction coords (targets)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']
        pred = mlp_fit(z, c, dev)
        del a, m, drop, E, Wd, c, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def setcfg(bf, GRP, corr=None, r=None, BR=None, PRED=None, neuron=False):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=True, B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=neuron, corr=corr, r=r, Wd=s['Wd'] if corr == 'oracle' else None,
                                Br=BR[li] if BR else None, xbar=s['xbar'], P=s['P'], pred=PRED[li] if PRED else None))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = None; CFG[li]['neuron'] = False

    print(f"  SwiGLU ppl (dense {dense_ppl:.3f}), keep-pattern K={K}, oracle group select\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        setcfg(bf, GRP); base = ppl(); off()
        setcfg(bf, GRP, neuron=True); nu = ppl(); off()
        report_metrics(dict(seed=SEED, keep=bf, dense=dense_ppl, group_oracle=base, neuron_oracle=nu))
        print(f"  keep{int(bf*100)}: group-oracle {base:.3f} | neu-oracle {nu:.3f} | dense {dense_ppl:.3f}", flush=True)
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = basis_and_pred(li, bf, *GRP[li])
        print(f"  {'r':>4} | {'+oracle-corr':>12} | {'+predicted-corr':>15}", flush=True)
        for r in RANKS:
            setcfg(bf, GRP, corr='oracle', r=r, BR=BR); oc = ppl(); off()
            setcfg(bf, GRP, corr='pred', r=r, BR=BR, PRED=PRED); pc = ppl(); off()
            print(f"  {r:>4} | {oc:>12.3f} | {pc:>15.3f}", flush=True)
            report_metrics(dict(seed=SEED, keep=bf, rank=r, true_coordinates=oc, predicted_coordinates=pc))
        print("", flush=True)
    print("READ: predicted-corr between group-oracle (no corr) and oracle-corr => how much of the low-rank", flush=True)
    print("output correction is RECOVERABLE from x (deployable). If predicted≈oracle => A is deployably correctable.", flush=True)


if __name__ == "__main__":
    main()
