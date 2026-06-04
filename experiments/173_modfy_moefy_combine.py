"""Experiment 173 — MoDfy x MoEfy: do the two ORTHOGONAL conditional-compute axes combine?
MoEfy = within-FFN neuron keep (oracle top-k per token). MoDfy = per-token whole-FFN SKIP
(skip lowest-‖FFN_out‖ tokens => pure residual). At MATCHED total FFN compute c=(1-skip)*keep,
compare: pure-MoEfy(keep=c,skip=0) vs pure-MoD(skip=1-c,keep=1) vs combined. If MoD/combined
beats pure-MoEfy at same compute => per-token SKIP axis is rich (unlike exp172's tiny neuron-budget
axis) => positioning "MoDfy+MoEfy" validated. ORACLE ceiling first. Qwen-0.5B.
Run: python3 experiments/173_modfy_moefy_combine.py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_CALIB = 2048
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
    N = a.shape[0]; Kc = mlp._gsz.shape[0]
    # --- MoEfy axis: ORACLE per-token neuron keep (route-all, top route_budget) ---
    if mlp._route_budget >= mlp._dff:                     # keep=1.0 => no MoE drop
        m = torch.ones(N, a.shape[1], device=a.device, dtype=a.dtype)
    else:
        sq = ((a.float() - mlp._mean.float()).abs() * mlp._vn) ** 2
        score = torch.zeros(N, Kc, device=a.device).index_add_(1, mlp._routed_grp_full, sq)
        order = score.argsort(1, descending=True); so = mlp._gsz[order]
        keep_ord = (so.cumsum(1) - so) < mlp._route_budget
        selg = torch.zeros(N, Kc, dtype=torch.bool, device=a.device)
        selg.scatter_(1, order, keep_ord)
        m = selg[:, mlp._routed_grp_full].to(a.dtype) * mlp._routed_is
    out = mlp.down_proj(a * m + mlp._rep * (1 - m))
    # --- MoDfy axis: per-token whole-FFN SKIP (oracle: lowest-||out|| tokens -> pure residual) ---
    nskip = int(round(mlp._skip * N))
    if nskip > 0:
        imp = out.float().norm(dim=1)
        thr = imp.kthvalue(nskip).values                  # nskip-th smallest
        tok_skip = imp <= thr
        out = out * (~tok_skip).unsqueeze(1).to(out.dtype)
        run = 1.0 - tok_skip.float().mean().item()
    else:
        run = 1.0
    mlp._compute = m.mean().item() * run                  # actual FFN compute fraction
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)

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
            routed_is = torch.zeros(dff, device=dev)
            for g in range(Kc):
                ix = routed_pool[(grp_local == g).nonzero().flatten()]
                gsz[g] = len(ix); grp_full[ix] = g; routed_is[ix] = 1.0
            vn = Wd.norm(dim=0).clone()
            del Wup, Wd, dev_a, contrib; gc.collect(); torch.cuda.empty_cache()
            mlp = layers[li].mlp
            mlp._xbar = s['xbar']; mlp._P = s['P']
            mlp._gsz = gsz.to(model.dtype); mlp._route_budget = float(route_budget)
            mlp._shared_mask = shared_mask.to(model.dtype).unsqueeze(0)
            mlp._routed_grp_full = grp_full; mlp._routed_is = routed_is.to(model.dtype)
            mlp._rep = mlp._mean; mlp._vn = vn; mlp._dff = dff; mlp._skip = 0.0
        for li in range(nL):
            layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    def set_skip(s):
        for li in range(nL):
            layers[li].mlp._skip = s

    def realcompute():
        return sum(layers[li].mlp._compute for li in range(nL)) / nL

    def ppl_at(keep, skip):
        configure(keep, 0.0); set_skip(skip)
        ce = ce_eval()
        return float(torch.tensor(ce).exp()), realcompute()

    dense_ce = ce_eval()
    print(f"\n  MoDfy (token-skip) x MoEfy (neuron-keep): matched-compute frontier, ORACLE")
    print(f"  dense Qwen ppl {torch.tensor(dense_ce).exp():.2f}\n")
    print(f"  {'target c':>8s} | {'recipe':>16s} | {'keep':>5s} {'skip':>5s} | {'real c':>6s} | {'ppl':>8s}")
    print("  " + "-" * 62)
    import math
    for c in [0.75, 0.5, 0.25]:
        # three ways to spend the same FFN compute fraction c = (1-skip)*keep
        recipes = [("pure-MoEfy", c, 0.0),
                   ("pure-MoD", 1.0, 1.0 - c),
                   ("combined", math.sqrt(c), 1.0 - math.sqrt(c))]
        for tag, keep, skip in recipes:
            p, rc = ppl_at(keep, skip)
            print(f"  {c:>8.2f} | {tag:>16s} | {keep:>5.2f} {skip:>5.2f} | {rc:>6.3f} | {p:8.2f}", flush=True)
        print("  " + "-" * 62)
    print("\nREAD: at matched real-c, pure-MoD or combined < pure-MoEfy => per-token SKIP axis is")
    print("rich (MoD captures token heterogeneity MoEfy can't) => 'MoDfy x MoEfy' positioning holds.")
    print("pure-MoEfy best => skip axis adds nothing here. ORACLE ceiling; learnability/distill next.")


if __name__ == "__main__":
    main()
