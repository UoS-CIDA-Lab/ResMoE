"""Experiment 234 — low-rank output correction on a NON-GATED CAUSAL decoder (ReLU): OPT-1.3b.
The correction is activation-agnostic: it captures a = act(W_in x) and reconstructs the dropped-output
error e = sum_{dropped}(a_k - abar_k) v_k from x, regardless of ReLU/GeLU/SwiGLU. OPT has fc1/fc2 (no gate),
so we capture x = fc1 input, a = fc2 input. Reports dense / neuron-oracle / group-oracle / +predicted-corr
at keep50 & keep25 (same protocol as 233), to populate the ReLU row of the multi-model correction table.
Run: HHMODEL=facebook/opt-1.3b python3 experiments/234_relu_causal.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.reproducibility import configure_determinism, report_provenance, report_metrics
import torch, torch.nn as nn, torch.nn.functional as F

SEED = int(os.environ.get("SEED", "0"))
configure_determinism(SEED)

MODEL = os.environ.get("HHMODEL", "facebook/opt-1.3b")
N_CALIB = int(os.environ.get("NCALIB", "8192"))
CHUNK = int(os.environ.get("CHUNK", "512"))
N_EVAL = 1024
K = 128
RCORR = 128
RFEAT = 512
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.50,0.25").split(",")]


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


def oracle_mask(a, abar, vn, gsz, gf, B):
    per = ((a - abar) * vn) ** 2
    sg = torch.zeros(a.shape[0], gsz.shape[0], device=a.device).index_add_(1, gf, per)
    return keep_topB_group(sg, gsz, gf, B).to(a.dtype)


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
    # locate decoder layers
    base = model.model if hasattr(model, "model") else model
    layers = base.decoder.layers if hasattr(base, "decoder") else base.layers
    nL = len(layers)
    inproj = []; outproj = []
    for li in range(nL):
        ip, op = find_ffn(layers[li]); inproj.append(ip); outproj.append(op)
    torch.set_grad_enabled(False)
    report_provenance(source=__file__, model=MODEL, revision=model.config._commit_hash,
                      seed=SEED, tokens=ids, fingerprints=fingerprints,
                      calibration_items=N_CALIB, evaluation_items=N_EVAL, chunk=CHUNK)

    QBITS = int(os.environ.get("QBITS", "0"))                 # 0=off; else fake-quant FFN weights (composability)
    if QBITS:
        qmax = 2 ** (QBITS - 1) - 1

        def fakequant(W):
            s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
            return ((W / s).round().clamp(-qmax, qmax) * s)
        nq = 0
        for li in range(nL):
            mlp = getattr(layers[li], "mlp", layers[li])
            for m in [inproj[li], outproj[li], getattr(mlp, "up_proj", None)]:
                if m is not None:
                    m.weight.data = fakequant(m.weight.data.float()).to(m.weight.dtype); nq += 1
        print(f"  fake-quantized {nq} FFN weight matrices to {QBITS}-bit (composability test)", flush=True)

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
        inproj[li].register_forward_pre_hook(gpre(li))
        outproj[li].register_forward_pre_hook(dpre(li))
        outproj[li].register_forward_hook(dpost(li))

    # ---- harvest ----
    capa: dict[int, list[torch.Tensor]] = {li: [] for li in range(nL)}; capx: dict[int, list[torch.Tensor]] = {li: [] for li in range(nL)}
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

    def ce_eval():
        tot = 0.0; n = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); n += tgt.numel()
        return torch.tensor(tot / n).exp().item()

    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li]).float().to(dev)
        Wd = outproj[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        af = a.float().to(dev); vn = Wd.norm(dim=0); abar = af.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a, x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        del capa[li], capx[li]
        del af, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  MODEL={MODEL} dff={dff} nL={nL} act={getattr(model.config,'activation_function',getattr(model.config,'hidden_act','?'))}", flush=True)

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

    def setcfg(bf, GRP, mode, Br=None, pred=None):
        B = int(round(bf * dff))
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update(dict(active=(mode != 'off'), B=B, vn=s['vn'], abar=s['abar'], gsz=gsz, gf=gf,
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                Br=(Br[li] if Br else None), xbar=s['xbar'], P=s['P'],
                                pred=(pred[li] if pred else None)))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    dense = ce_eval(); report_metrics(dict(seed=SEED, configuration="dense", ppl=dense)); print(f"  dense ppl {dense:.3f}\n", flush=True)
    for bf in KEEPS:
        GRP = {li: grouping(li, bf) for li in range(nL)}
        setcfg(bf, GRP, 'group'); floor = ce_eval(); off()
        setcfg(bf, GRP, 'neuron'); neu = ce_eval(); off()
        BR = {}; PRED = {}
        for li in range(nL):
            BR[li], PRED[li] = basis_and_pred(li, bf, *GRP[li])
        setcfg(bf, GRP, 'corr', BR, PRED); corr = ce_eval(); off()
        report_metrics(dict(seed=SEED, keep=bf, qbits=QBITS, dense=dense, group_oracle=floor, neuron_oracle=neu, corrected=corr))
        print(f"  keep{int(bf*100)}: dense {dense:.3f} | neu-oracle {neu:.3f} | group-oracle {floor:.3f} | +corr {corr:.3f}", flush=True)
    print("\nREAD: ReLU (OPT) row for the multi-model correction table; corr<group-oracle => correction generalizes to ReLU.", flush=True)


if __name__ == "__main__":
    main()
