"""Experiment 160 — DOWNSTREAM TASK accuracy (lm-eval) for shared+routed vs route-all G-MoE,
Phi-2 (G-MoE baseline). Tasks: boolq, copa, rte (loglikelihood, fast). Configs: dense /
route-all(sf0) / shared+rt(sf60) at keep 50/25. NOTE Phi-2 is the perplexity-EXCEPTION (exp157:
route-all doesn't collapse), so this tests whether TASK ACCURACY tracks the perplexity finding
(=> validates perplexity as task proxy for the mBERT/SantaCoder wins). Run: python3 .../160_...py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_CALIB = 2048
CHUNK = 512
KROUTE = 64
TASKS = ["boolq", "piqa", "arc_easy"]


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
    a = mlp.act_fn(mlp.gate_proj(xf)) * mlp.up_proj(xf)   # SwiGLU hidden
    z = (xf.float() - mlp._xbar) @ mlp._P
    score = mlp._router(z).to(a.dtype)
    order = score.argsort(1, descending=True); so = mlp._gsz[order]
    keep_ord = (so.cumsum(1) - so) < mlp._route_budget
    selg = torch.zeros(a.shape[0], mlp._gsz.shape[0], dtype=torch.bool, device=a.device)
    selg.scatter_(1, order, keep_ord)
    m = mlp._shared_mask.expand(a.shape[0], -1).clone()
    m = m + selg[:, mlp._routed_grp_full].to(m.dtype) * mlp._routed_is
    m = m.clamp(max=1.0)
    out = mlp.down_proj(a * m + mlp._rep * (1 - m))
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(dev).eval()  # fp32 for stable fine-tune
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)

    def run_eval(tag):
        torch.set_grad_enabled(False)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=4)
        r = lm_eval.simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, verbosity="ERROR")
        accs = {t: r['results'][t].get('acc,none', r['results'][t].get('acc')) for t in TASKS}
        avg = sum(accs.values()) / len(accs)
        print(f"  [{tag}] " + "  ".join(f"{t}={accs[t]:.3f}" for t in TASKS) + f"  | avg={avg:.4f}", flush=True)
        return avg

    print("evaluating DENSE phi-2 ...", flush=True)
    base = run_eval("dense")

    # harvest calib activations for router building
    torch.set_grad_enabled(False)
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        mlp = layers[li].mlp
        hs.append(mlp.gate_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
        hs.append(mlp.down_proj.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    LST = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        Wup = layers[li].mlp.gate_proj.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        Wd = layers[li].mlp.down_proj.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        abar = a.mean(0); dev_a = a - abar
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False); P = Vt[:128].T
        contrib = dev_a.abs() * Wd.norm(dim=0)
        LST[li] = dict(Wup=Wup.cpu(), Wd=Wd.cpu(), dev_a=dev_a.cpu(), contrib=contrib.cpu(),
                       Z=((x - xbar) @ P).float(), xbar=xbar.float(), P=P.float())
        layers[li].mlp._mean = abar.to(model.dtype)
        del x, a, dev_a, contrib, Wup, Wd; gc.collect(); torch.cuda.empty_cache()
    capx.clear(); capa.clear(); gc.collect(); torch.cuda.empty_cache()
    tr = slice(0, N_CALIB)

    def configure(bf, sf):
        B = int(round(bf * dff))
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
            router = mlp_fit(s['Z'][tr], rn[tr], dev)
            del Wup, Wd, dev_a, contrib, rn; gc.collect(); torch.cuda.empty_cache()
            mlp = layers[li].mlp
            mlp._xbar = s['xbar']; mlp._P = s['P']; mlp._router = router
            mlp._gsz = gsz.to(model.dtype); mlp._route_budget = float(route_budget)
            mlp._shared_mask = shared_mask.to(model.dtype).unsqueeze(0)
            mlp._routed_grp_full = grp_full; mlp._routed_is = routed_is.to(model.dtype)
            mlp._rep = mlp._mean
        for li in range(nL):
            layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    import copy
    sd0 = copy.deepcopy({k: v for k, v in model.state_dict().items()})
    FT_STEPS = 300

    def finetune():
        for p in model.parameters():
            p.requires_grad_(False)
        ftp = []
        for li in range(nL):
            mlp = layers[li].mlp
            for mod in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                mod.weight.requires_grad_(True); ftp.append(mod.weight)
            mlp._rep = mlp._mean.clone().requires_grad_(True); ftp.append(mlp._rep)
        opt = torch.optim.AdamW(ftp, lr=1e-5)
        gen = torch.Generator(device=dev).manual_seed(1)
        torch.set_grad_enabled(True)
        for st in range(FT_STEPS):
            c0 = int(torch.randint(0, N_CALIB - CHUNK, (1,), generator=gen, device=dev).item())
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            opt.zero_grad()
            lo = model(xx).logits[0, :-1].float()
            loss = F.cross_entropy(lo, ids[c0 + 1:c0 + CHUNK].to(dev))
            loss.backward(); torch.nn.utils.clip_grad_norm_(ftp, 1.0); opt.step()
        torch.set_grad_enabled(False)
        for p in ftp:
            p.grad = None
            p.requires_grad_(False)
        del opt; gc.collect(); torch.cuda.empty_cache()

    print(f"\n  PRACTICAL REGIME + FINE-TUNE: task acc (dense {base:.4f}); route-all=G-MoE full recipe\n")
    for bf in [0.85, 0.5]:
        for sf, tagc in [(0.0, "route-all+ft"), (0.6, "shared+rt+ft")]:
            model.load_state_dict(sd0, strict=False)
            configure(bf, sf)
            finetune()
            run_eval(f"keep{int(bf*100)}-{tagc}")
    print("\nREAD: shared+rt+ft > route-all+ft at keep85 (practical) on TASK ACC, vs G-MoE full")
    print("recipe (trained router + rep + fine-tune) => passes validity gates #1 (full baseline) + #2")
    print("(practical regime). ~equal/worse => the win doesn't survive a strong baseline + fine-tune.")


if __name__ == "__main__":
    main()
