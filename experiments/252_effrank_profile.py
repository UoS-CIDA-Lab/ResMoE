"""Experiment 252 — EFFECTIVE-RANK profile of the dropped-output error E, per layer (depth).
The low-rank correction reconstructs E = drop @ Wd.T (output space, [N, d]) with a rank-r SVD basis Br.
234/235 use a FIXED r (=RCORR=128) for every layer. To allocate r per depth we must first MEASURE how
hard each layer's E is to reconstruct linearly. Unlike 221 (which spectra the INPUT-space contribution /
keep-indicator), this script spectra the OUTPUT-space E that the correction actually rebuilds, using the
SAME group-oracle drop as the correction. Per layer & keep we report energy-threshold rank r@tau, the
entropy effective rank (Roy-Vetterli), and the stable rank; we print a depth table, save a PNG profile,
and dump the per-layer spectra for the budget-redistribution step (234 RANKMODE=budget).
Run: HHMODEL=facebook/opt-1.3b python3 experiments/252_effrank_profile.py
"""
from __future__ import annotations
import sys, pathlib, gc, os, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.reproducibility import load_mbpp_code
import torch

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = int(os.environ.get("NCALIB", "8192"))
CHUNK = int(os.environ.get("CHUNK", "512"))
K = int(os.environ.get("K", "128"))
RFEAT = int(os.environ.get("RFEAT", "512"))               # x-feature PCA dim for the predictable-rank measure
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.50,0.25").split(",")]
TAUS = [float(x) for x in os.environ.get("TAUS", "0.9,0.95,0.99").split(",")]
OUTDIR = os.environ.get("OUTDIR", ".")


# ---- helpers copied from 234 (scripts are self-contained per repo convention) ----
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


def find_ffn(layer):
    """Return (in_proj, out_proj) modules for a causal block, gated or not."""
    mlp = getattr(layer, "mlp", layer)
    if hasattr(mlp, "gate_proj"):                 # SwiGLU (Qwen/Llama)
        return mlp.gate_proj, mlp.down_proj
    if hasattr(mlp, "fc1"):                        # OPT-style
        return mlp.fc1, mlp.fc2
    if hasattr(layer, "fc1"):                      # OPT puts fc1/fc2 on the layer
        return layer.fc1, layer.fc2
    if hasattr(mlp, "c_fc"):                        # GPT-2 style
        return mlp.c_fc, mlp.c_proj
    raise RuntimeError("unknown FFN structure")

# ---- effective-rank metrics from a singular-value spectrum ----
def rank_metrics(S, taus):
    """S: 1D descending singular values (>=0). Returns (r@tau dict, entropy effective rank, stable rank)."""
    s2 = S * S
    energy = s2.cumsum(0) / s2.sum().clamp(min=1e-30)
    r_tau = {t: int((energy < t).sum().item()) + 1 for t in taus}
    p = (S / S.sum().clamp(min=1e-30)).clamp(min=1e-30)
    erank = float(torch.exp(-(p * p.log()).sum()).item())          # Roy-Vetterli effective rank
    srank = float((s2.sum() / s2[0].clamp(min=1e-30)).item())      # ||E||_F^2 / sigma_max^2
    return r_tau, erank, srank


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = load_mbpp_code()
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    base = model.model if hasattr(model, "model") else model
    layers = base.decoder.layers if hasattr(base, "decoder") else base.layers
    nL = len(layers)
    inproj = []; outproj = []
    for li in range(nL):
        ip, op = find_ffn(layers[li]); inproj.append(ip); outproj.append(op)
    torch.set_grad_enabled(False)

    # ---- harvest the FFN-intermediate activation a (out-proj input) AND the FFN input x (in-proj input) ----
    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(outproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(inproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    act = getattr(model.config, "activation_function", getattr(model.config, "hidden_act", "?"))
    print(f"  MODEL={MODEL}  dff={dff}  nL={nL}  act={act}  K={K}  RFEAT={RFEAT}  N_CALIB={N_CALIB}", flush=True)

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wd = outproj[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, abar=af.mean(0), vn=Wd.norm(dim=0), Wd=Wd.cpu(),
                       x=x.half().cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = None; capx[li] = None; del af, x; gc.collect(); torch.cuda.empty_cache()

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

    def error_spectrum(li, bf, gsz, gf):
        """Return (S_raw, S_pred): singular spectra of the group-oracle dropped-output error E and of its
        x-PREDICTABLE part Ê. Ê = z (z⁺E) is the OLS projection of E onto the FFN-input feature space
        z=(x-x̄)P (same RRR subspace as 233); its rank bounds what a deployable predictor can reconstruct,
        so it — not raw E — is the decision-relevant rank for allocating the correction budget."""
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev)
        B = int(round(bf * dff))
        m = oracle_mask(a, abar, s['vn'], gsz, gf, B)
        E = (a - (a * m + abar * (1 - m))) @ Wd.T          # [N, d] group-oracle dropped output error
        S_raw = torch.linalg.svdvals(E).detach().cpu()
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']  # [N, RFEAT] FFN-input features
        coef = torch.linalg.lstsq(z, E).solution           # [RFEAT, d] OLS  E ~ z
        Ehat = z @ coef                                     # x-predictable part of E (rank <= RFEAT)
        S_pred = torch.linalg.svdvals(Ehat).detach().cpu()
        del a, m, E, Wd, z, coef, Ehat; gc.collect(); torch.cuda.empty_cache()
        return S_raw, S_pred

    # ---- measure per layer & keep: raw E vs x-predictable Ê ----
    DUMP = {"model": MODEL, "nL": nL, "dff": dff, "rfeat": RFEAT, "keeps": KEEPS, "taus": TAUS, "spectra": {}}
    PROF = {}  # bf -> {'raw': {...}, 'pred': {...}}
    t95 = 0.95 if 0.95 in TAUS else TAUS[len(TAUS) // 2]
    for bf in KEEPS:
        kp = int(bf * 100)
        R = {'rt': {t: [] for t in TAUS}, 'er': [], 'sr': [], 'spec': []}
        Pr = {'rt': {t: [] for t in TAUS}, 'er': [], 'sr': [], 'spec': []}
        for li in range(nL):
            gsz, gf = grouping(li, bf)
            S_raw, S_pred = error_spectrum(li, bf, gsz, gf)
            for S, D in [(S_raw, R), (S_pred, Pr)]:
                r_tau, erank, srank = rank_metrics(S, TAUS)
                for t in TAUS:
                    D['rt'][t].append(r_tau[t])
                D['er'].append(erank); D['sr'].append(srank); D['spec'].append(S.half())
        DUMP["spectra"][kp] = {'raw': R['spec'], 'pred': Pr['spec']}
        PROF[bf] = {'raw': R, 'pred': Pr}
        print(f"\n  === keep{kp}: per-layer effective rank — E (raw) vs Ê (x-predictable, RFEAT={RFEAT}) ===", flush=True)
        print(f"  layer  E:r{t95:g}  E:erank  E:sr   |  Ehat:r{t95:g}  Ehat:erank  Ehat:sr", flush=True)
        for li in range(nL):
            print(f"  {li:5d}  {R['rt'][t95][li]:6d}  {R['er'][li]:7.1f}  {R['sr'][li]:5.1f}   |  "
                  f"{Pr['rt'][t95][li]:8d}  {Pr['er'][li]:9.1f}  {Pr['sr'][li]:6.1f}", flush=True)
        mn = lambda v: sum(v) / nL
        print(f"  mean   {mn(R['rt'][t95]):6.1f}  {mn(R['er']):7.1f}  {mn(R['sr']):5.1f}   |  "
              f"{mn(Pr['rt'][t95]):8.1f}  {mn(Pr['er']):9.1f}  {mn(Pr['sr']):6.1f}", flush=True)

    # ---- dump spectra for the budget-redistribution step ----
    slug = MODEL.rstrip("/").split("/")[-1]
    dump_path = os.path.join(OUTDIR, f"effrank_{slug}.pt")
    torch.save(DUMP, dump_path)
    print(f"\n  dumped per-layer spectra -> {dump_path}", flush=True)

    # ---- PNG profile (guarded: matplotlib is optional) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(KEEPS), figsize=(5.2 * len(KEEPS), 3.8), squeeze=False)
        x = list(range(nL))
        for j, bf in enumerate(KEEPS):
            ax = axes[0][j]; R = PROF[bf]['raw']; Pr = PROF[bf]['pred']
            ax.plot(x, R['rt'][t95], "-o", ms=3, color="C0", label=f"E  r@{t95:g}")
            ax.plot(x, R['er'], "-s", ms=3, color="C1", label="E  erank")
            ax.plot(x, Pr['rt'][t95], "--o", ms=3, color="C2", label=f"Ê  r@{t95:g} (x-pred)")
            ax.plot(x, Pr['er'], "--s", ms=3, color="C3", label="Ê  erank (x-pred)")
            ax.set_title(f"keep{int(bf*100)}"); ax.set_xlabel("layer (depth)")
            ax.set_ylabel("effective rank"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.suptitle(f"{slug}: per-layer effective rank — raw E vs x-predictable Ê (RFEAT={RFEAT})")
        fig.tight_layout()
        png_path = os.path.join(OUTDIR, f"effrank_{slug}_{N_CALIB}.png")
        fig.savefig(png_path, dpi=130); plt.close(fig)
        print(f"  saved depth profile -> {png_path}", flush=True)
    except ImportError:
        print("  [warn] matplotlib not installed; skipped PNG (text table + .pt dump still produced).", flush=True)

    print(f"\nREAD: the DEPLOYABLE correction is bounded by Ê (x-predictable), not raw E. If Ê's effective rank is", flush=True)
    print("far below raw E and varies across depth, depth-adaptive budget allocation (253 RANKMODE=budget) helps;", flush=True)
    print("if Ê's profile is flat, uniform RCORR is already near-optimal. Compare the two curves in the PNG.", flush=True)


if __name__ == "__main__":
    main()
