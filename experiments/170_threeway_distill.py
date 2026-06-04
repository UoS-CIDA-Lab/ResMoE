"""Experiment 157 — CAUSAL model-level confirmation of shared+routed (Phi-2, their GeLU baseline).
Patch ALL 32 FFNs: route-all (sh0=G-MoE) vs shared+routed (sh60) at keep 50/25. Metric = held-out
PERPLEXITY (the causal-LM task) + dense reference. If shared+rt ppl < route-all ppl at model level
on a CAUSAL model => the win generalizes beyond mBERT/MLM. Run: python3 experiments/157_...py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_CALIB = 8192
N_EVAL = 2048
CHUNK = 512
KROUTE = 64


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
    Kc = mlp._gsz.shape[0]
    if mlp._mode == "oracle":                            # true per-group residual norm (agg-exact proxy)
        sq = ((a.float() - mlp._mean.float()).abs() * mlp._vn) ** 2
        score = torch.zeros(a.shape[0], Kc, device=a.device).index_add_(1, mlp._routed_grp_full, sq).to(a.dtype)
    else:
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
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)
    dense_ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(dev).eval()
    for p in dense_ref.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def ce_eval():
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return tot / ntok

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
    print(f"Qwen2.5-0.5B: {nL} layers, dff={dff}", flush=True)
    LST = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        Wup = layers[li].mlp.gate_proj.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        Wd = layers[li].mlp.down_proj.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        abar = a.mean(0); dev_a = a - abar
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False); P = Vt[:128].T
        contrib = dev_a.abs() * Wd.norm(dim=0)
        LST[li] = dict(Wup=Wup.cpu(), Wd=Wd.cpu(), abar=abar, dev_a=dev_a.cpu(), contrib=contrib.cpu(),
                       Z=((x - xbar) @ P).float(), xbar=xbar.float(), P=P.float())
        layers[li].mlp._mean = abar.to(model.dtype)
        del x, a, dev_a, contrib, Wup, Wd; gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0:
            print(f"  prepped {li}", flush=True)
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
            vn = Wd.norm(dim=0).clone()
            del Wup, Wd, dev_a, contrib, rn; gc.collect(); torch.cuda.empty_cache()
            mlp = layers[li].mlp
            mlp._xbar = s['xbar']; mlp._P = s['P']; mlp._router = router
            mlp._gsz = gsz.to(model.dtype); mlp._route_budget = float(route_budget)
            mlp._shared_mask = shared_mask.to(model.dtype).unsqueeze(0)
            mlp._routed_grp_full = grp_full; mlp._routed_is = routed_is.to(model.dtype)
            mlp._rep = mlp._mean; mlp._vn = vn; mlp._mode = "deployable"
        for li in range(nL):
            layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    def set_mode(mode):
        for li in range(nL):
            layers[li].mlp._mode = mode

    import copy
    sd0 = copy.deepcopy(model.state_dict())
    FT_STEPS = 1500

    def finetune(loss_mode):                       # 'ce' or 'distill'
        for p in model.parameters():
            p.requires_grad_(False)
        ftp = []
        for li in range(nL):
            mlp = layers[li].mlp
            for mod in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                mod.weight.requires_grad_(True); ftp.append(mod.weight)
            mlp._rep = mlp._mean.clone().requires_grad_(True); ftp.append(mlp._rep)
        opt = torch.optim.AdamW(ftp, lr=2e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-5, total_steps=FT_STEPS, pct_start=0.1)
        gen = torch.Generator(device=dev).manual_seed(1)
        torch.set_grad_enabled(True)
        for st in range(FT_STEPS):
            c0 = int(torch.randint(0, N_CALIB - CHUNK, (1,), generator=gen, device=dev).item())
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            opt.zero_grad()
            sl = model(xx).logits[0, :-1].float()
            if loss_mode == "distill":
                with torch.no_grad():
                    dl = dense_ref(xx).logits[0, :-1].float()
                loss = F.kl_div(F.log_softmax(sl, -1), F.softmax(dl, -1), reduction='batchmean')
            else:
                loss = F.cross_entropy(sl, ids[c0 + 1:c0 + CHUNK].to(dev))
            loss.backward(); torch.nn.utils.clip_grad_norm_(ftp, 1.0); opt.step(); sched.step()
        torch.set_grad_enabled(False)
        for p in ftp:
            p.grad = None; p.requires_grad_(False)
        del opt; gc.collect(); torch.cuda.empty_cache()

    dense_ce = ce_eval()
    print(f"\n  3-WAY + STRONG DISTILL. dense ppl {torch.tensor(dense_ce).exp():.2f}\n")
    print(f"  {'keep':>5s} | {'config':>14s} | {'static':>8s} | {'distill-ft':>10s}")
    print("  " + "-" * 46)
    cfgs = [(0.0, "full-routing"), (0.6, "shared+rt"), (1.0, "full-sharing")]
    for bf in [0.85, 0.5]:
        for sf, tagc in cfgs:
            model.load_state_dict(sd0, strict=False); configure(bf, sf); set_mode("deployable")
            p_static = float(torch.tensor(ce_eval()).exp())
            model.load_state_dict(sd0, strict=False); configure(bf, sf); set_mode("deployable")
            finetune("distill"); p_kd = float(torch.tensor(ce_eval()).exp())
            print(f"  {int(bf*100):>4d}% | {tagc:>14s} | {p_static:8.2f} | {p_kd:10.2f}", flush=True)
        print("  " + "-" * 46)
    print("\nREAD: after the SAME strong distill, does shared+rt still beat full-routing AND full-sharing?")
    print("If distill closes the gap (route-all-distill ~ shared-distill) => shared advantage erased by")
    print("fine-tune. If shared-distill still < both => the advantage is robust to fine-tune.")


if __name__ == "__main__":
    main()
