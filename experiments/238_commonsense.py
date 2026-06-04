"""Experiment 238 — DOWNSTREAM (commonsense zero-shot) verify of the low-rank correction.
Standard multi-task suite used by efficiency papers: ARC-Easy/Challenge, PIQA, HellaSwag, Winogrande
(all cached). Qwen2.5-1.5B (general), keep-pattern K128, keep50. Accuracy for:
dense | group-oracle (no corr) | group-oracle + predicted rank-128 correction | neu-oracle (ceiling).
Run: HHMODEL=Qwen/Qwen2.5-1.5B python3 experiments/238_commonsense.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-1.5B")
N_CALIB = 8192
CHUNK = 512
K = 128
RCORR = 128
BF = 0.50


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
    CALIB = os.environ.get("CALIB", "wikitext")               # in-domain (natural language) for commonsense
    if CALIB == "wikitext":
        wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        buf = []
        for t in wt["text"]:
            if t.strip():
                buf.append(t)
            if len(buf) >= 20000:
                break
        ids = tok("\n\n".join(buf), return_tensors="pt").input_ids[0]
    else:
        code = []
        for sp in ["train", "test", "validation", "prompt"]:
            try:
                code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
            except Exception:
                pass
        ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    print(f"  calibration corpus = {CALIB} ({ids.shape[0]} tokens)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    gproj = [layers[li].mlp.gate_proj for li in range(nL)]
    downs = [layers[li].mlp.down_proj for li in range(nL)]
    torch.set_grad_enabled(False)

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
            x = XC[li].float(); sh = output.shape
            ehat = cfg['pred']((x - cfg['xbar']) @ cfg['P']) @ cfg['Br'].T
            return (output.reshape(-1, sh[-1]) + ehat.to(output.dtype)).reshape(sh)
        return hook
    for li in range(nL):
        gproj[li].register_forward_pre_hook(gpre(li))
        downs[li].register_forward_pre_hook(dpre(li))
        downs[li].register_forward_hook(dpost(li))

    # ---- harvest calib activations, build grouping + correction ----
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
    print(f"  MODEL={MODEL} dff={dff}; building grouping+correction @keep{int(BF*100)}", flush=True)

    GS = {}; GF = {}; ABAR = {}; VN = {}; XBAR = {}; P = {}; BR = {}; PRED = {}
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev); x = torch.cat(capx[li]).float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0); abar = a.mean(0); B = int(round(BF * dff))
        Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
        w = ((a - abar).abs() * vn).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=0))
        gsz = torch.zeros(K, device=dev)
        for g in range(K):
            gsz[g] = (gl == g).sum()
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False); Pm = VtX[:512].T
        m = oracle_mask(a, abar, vn, gsz, gl, B)
        drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T; _, _, Vt = torch.linalg.svd(E, full_matrices=False); Br = Vt[:RCORR].T
        pred = mlp_fit((x - xbar) @ Pm, E @ Br, dev)
        GS[li] = gsz; GF[li] = gl; ABAR[li] = abar; VN[li] = vn; XBAR[li] = xbar; P[li] = Pm; BR[li] = Br; PRED[li] = pred
        capa[li] = None; capx[li] = None
        del a, x, Wd, Bind, m, drop, E, Vt; gc.collect(); torch.cuda.empty_cache()
    print("  built. running HumanEval...\n", flush=True)

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    TASKS = os.environ.get("TASKS", "arc_easy,piqa,hellaswag,winogrande").split(",")

    def gen_eval(tag):
        model.config.use_cache = True
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
        accs = {}
        for t in TASKS:                                          # per-task so one offline failure doesn't kill all
            try:
                r = lm_eval.simple_evaluate(model=lm, tasks=[t], num_fewshot=0, verbosity="ERROR")
                res = r['results'][t]; accs[t] = res.get('acc_norm,none', res.get('acc,none'))
                del r
            except Exception as e:
                print(f"    [skip {t}] {str(e)[:60]}", flush=True)
        mean = (sum(accs.values()) / len(accs)) if accs else float('nan')
        line = " ".join(f"{t.split('_')[0]}={accs[t]:.3f}" for t in accs)
        print(f"  [{tag}] {line} | MEAN={mean:.3f}", flush=True)
        model.config.use_cache = False
        del lm; gc.collect(); torch.cuda.empty_cache()
        return mean

    def setcfg(mode):
        B = int(round(BF * dff))
        for li in range(nL):
            on = (mode != 'dense')
            CFG[li].update(dict(active=on, B=B, abar=ABAR[li], vn=VN[li], gsz=GS[li], gf=GF[li],
                                neuron=(mode == 'neuron'), corr=(mode == 'corr'),
                                xbar=XBAR[li], P=P[li], Br=BR[li], pred=PRED[li]))

    setcfg('dense'); gen_eval("dense")
    setcfg('oracle'); gen_eval(f"group-oracle keep{int(BF*100)} (no corr)")
    setcfg('corr'); gen_eval(f"group-oracle keep{int(BF*100)} + corr r{RCORR}")
    setcfg('neuron'); gen_eval(f"neu-oracle keep{int(BF*100)} (ceiling)")
    print("\nREAD: if +corr pass@1 > group-oracle pass@1 => the low-rank correction's ppl gain carries to", flush=True)
    print("the downstream code-generation task (not just perplexity).", flush=True)


if __name__ == "__main__":
    main()
