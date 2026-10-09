"""Experiment 247 — BEHAVIORAL PRESERVATION (SE equivalence-testing lens).
How faithfully does the modularized FFN (+ correction module) reproduce the ORIGINAL dense model's
outputs? Metrics vs dense: top-1 next-token agreement (%) and mean KL(dense || modular). For
group-oracle (no corr) | +correction | neuron-oracle, keep50. Higher agreement / lower KL = better
behavior preservation. Qwen2.5-Coder-1.5B (SwiGLU). Run: python3 experiments/247_behavior.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192; CHUNK = 512; N_EVAL = 1024
K = 128; RCORR = 128; RFEAT = 512
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


def oracle_mask(a, abar, vn, gsz, gf, B):
    per = ((a - abar) * vn) ** 2
    sg = torch.zeros(a.shape[0], gsz.shape[0], device=a.device).index_add_(1, gf, per)
    return keep_topB_group(sg, gsz, gf, B).to(a.dtype)


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
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    torch.set_grad_enabled(False)
    CFG = {li: {'active': False} for li in range(nL)}; XC = {li: None for li in range(nL)}

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
                m = keep_topB_neuron((af.float() - cfg['abar']).abs() * cfg['vn'], cfg['B']).to(a.dtype)
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
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    capa = {li: [] for li in range(nL)}; capx = {li: [] for li in range(nL)}; hs = []
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

    GS = {}; GF = {}; ABAR = {}; VN = {}; XBAR = {}; P = {}; BR = {}; PRED = {}
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        STR_a = a; STR_x = x; XBAR[li] = x.mean(0)
        _, _, VtX = torch.linalg.svd(x - XBAR[li], full_matrices=False); P[li] = VtX[:RFEAT].T
        ABAR[li] = abar; VN[li] = vn
        STR_a = STR_a; capa[li] = (a.half().cpu()); capx[li] = (x.half().cpu())
        layers[li]._Wd = Wd.cpu()
        del a, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  MODEL={MODEL} dff={dff} (behavioral preservation vs dense)", flush=True)

    def build(bf):
        B = int(round(bf * dff))
        for li in range(nL):
            a = capa[li].float().to(dev); abar = ABAR[li]; vn = VN[li]
            Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
            w = ((a - abar).abs() * vn).mean(0)
            gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
            gsz = torch.zeros(K, device=dev)
            for g in range(K):
                gsz[g] = (gl == g).sum()
            Wd = layers[li]._Wd.to(dev)
            m = oracle_mask(a, abar, vn, gsz, gl, B); drop = a - (a * m + abar * (1 - m))
            E = drop @ Wd.T; Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
            z = (capx[li].float().to(dev) - XBAR[li]) @ P[li]; pred = mlp_fit(z, E @ Br, dev)
            GS[li] = gsz; GF[li] = gl; BR[li] = Br; PRED[li] = pred
            del a, Bind, m, drop, E, Ec, Vt, z, Wd; gc.collect(); torch.cuda.empty_cache()

    def setcfg(mode, B):
        for li in range(nL):
            CFG[li].update(dict(active=(mode != 'dense'), B=B, vn=VN[li], abar=ABAR[li], gsz=GS.get(li), gf=GF.get(li),
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                Br=BR.get(li), xbar=XBAR[li], P=P[li], pred=PRED.get(li)))

    def off():
        for li in range(nL):
            CFG[li]['active'] = False; CFG[li]['corr'] = False; CFG[li]['neuron'] = False

    def logits_pass():
        out = []
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            out.append(model(xx).logits[0].float().cpu())
        return torch.cat(out)                                  # [N_EVAL, V]
    setcfg('dense', 0); LD = logits_pass(); off()
    am_d = LD.argmax(-1); lp_d = F.log_softmax(LD, -1)

    def fidelity(mode, B):
        setcfg(mode, B); LM = logits_pass(); off()
        am_m = LM.argmax(-1)
        agree = (am_m == am_d).float().mean().item()
        lp_m = F.log_softmax(LM, -1)
        kl = (lp_d.exp() * (lp_d - lp_m)).sum(-1).mean().item()
        return agree, kl

    for bf in KEEPS:
        build(bf); B = int(round(bf * dff))
        gA, gK = fidelity('group', B)
        cA, cK = fidelity('corr', B)
        nA, nK = fidelity('neuron', B)
        print(f"  keep{int(bf*100)} top-1 agreement vs dense: group {gA:.3f} | +corr {cA:.3f} | neuron {nA:.3f}", flush=True)
        print(f"  keep{int(bf*100)} mean KL(dense||mod):       group {gK:.3f} | +corr {cK:.3f} | neuron {nK:.3f}", flush=True)
    print("\nREAD: +corr should raise top-1 agreement and lower KL vs group-oracle => the correction module", flush=True)
    print("makes the modularized FFN behave more like the original (behavior preservation).", flush=True)


if __name__ == "__main__":
    main()
