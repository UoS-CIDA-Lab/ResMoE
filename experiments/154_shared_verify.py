"""Experiment 153 — Does the shared+routed win (exp152, single-layer) SURVIVE at MODEL level?
Critical test: finer-K looked good single-layer but died at model level (exp142) because the
routing gap compounds. Shared neurons have ZERO routing error (always-on, exact) -> should NOT
compound -> should survive. mBERT, ALL 12 FFNs patched with shared+routed at matched budget
(keep 50%), deployable router on the routed pool. Measure MODEL-level MLM-CE: shared% 0 vs 30/60/90.
shared>0 << shared=0 at model level => a real beyond-G-MoE construction. Run: python3 .../153_...py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

NAME = "bert-base-multilingual-cased"
N_CALIB = 16384
N_EVAL = 8192
SEQ = 512
BUDGET = 0.5
SHARED_FRAC = [0.0, 0.3, 0.6, 0.9]
KROUTE = 64
MASK_P = 0.15


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


def mlp_fit(X, Y, dev, steps=1000, lr=3e-3, bs=2048, hidden=128, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def patched_intermediate(self, hidden_states):
    sh = hidden_states.shape
    xf = hidden_states.reshape(-1, sh[-1])
    a = self.intermediate_act_fn(self.dense(xf))          # [N, dff]
    z = (xf.float() - self._xbar) @ self._P
    score = self._router(z).to(a.dtype)                   # [N, Kc] routed groups
    order = score.argsort(1, descending=True)
    so = self._gsz[order]
    keep_ord = (so.cumsum(1) - so) < self._route_budget
    selg = torch.zeros(a.shape[0], self._gsz.shape[0], dtype=torch.bool, device=a.device)
    selg.scatter_(1, order, keep_ord)
    m = self._shared_mask.expand(a.shape[0], -1).clone()           # shared always 1
    m = m + selg[:, self._routed_grp_full].to(m.dtype) * self._routed_is        # add routed-kept
    m = m.clamp(max=1.0)
    out = a * m + self._mean * (1 - m)                             # gated activation (BertOutput.dense applies after)
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids_all = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
    mask_id = tok.mask_token_id
    model = AutoModelForMaskedLM.from_pretrained(NAME).to(dev).eval()
    layers = model.bert.encoder.layer; nL = len(layers)

    def batches(start, n):
        return [ids_all[c0:c0 + SEQ].unsqueeze(0).to(dev) for c0 in range(start, start + n, SEQ)]

    @torch.no_grad()
    def mlm_ce(seed=0):
        gen = torch.Generator(device=dev).manual_seed(seed); tot = 0.0; ntok = 0
        for xb in batches(N_CALIB, N_EVAL):
            inp = xb.clone()
            pm = (torch.rand(inp.shape, generator=gen, device=dev) < MASK_P); pm[:, 0] = False
            lab = inp.clone(); lab[~pm] = -100; inp[pm] = mask_id
            lo = model(inp).logits[0]
            tot += F.cross_entropy(lo.float(), lab[0], ignore_index=-100, reduction='sum').item()
            ntok += (lab[0] != -100).sum().item()
        return tot / max(ntok, 1)

    torch.set_grad_enabled(False)
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        inter = layers[li].intermediate
        hs.append(inter.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
        hs.append(inter.register_forward_hook(
            (lambda li: (lambda _m, _i, o: capa[li].append(o.reshape(-1, o.shape[-1]).float())))(li)))
    for xb in batches(0, N_CALIB):
        model(xb)
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]; B = int(round(BUDGET * dff))
    print(f"mBERT: {nL} layers, dff={dff}, budget {B}", flush=True)

    # precompute per-layer stats once
    LST = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        Wup = layers[li].intermediate.dense.weight.detach().float().to(dev)
        Wd = layers[li].output.dense.weight.detach().float().to(dev)
        abar = a.mean(0); dev_a = a - abar
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False); P = Vt[:128].T
        contrib = dev_a.abs() * Wd.norm(dim=0)
        topB = contrib.argsort(1, descending=True)[:, :B]
        freq = torch.zeros(dff, device=dev); freq.scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev))
        LST[li] = dict(x=x, a=a, Wup=Wup, abar=abar, xbar=xbar, P=P, freq=freq,
                       dev_a=dev_a, Wd=Wd)
        layers[li].intermediate.dense_out = layers[li].output.dense   # alias for patched fwd
        layers[li].intermediate._mean = abar.to(model.dtype)
        if li % 4 == 0:
            print(f"  prepped layer {li}", flush=True)

    Z = {li: ((LST[li]['x'] - LST[li]['xbar']) @ LST[li]['P']).float() for li in range(nL)}
    tr = slice(0, 10240)

    def configure(sf, seed, mode):
        for li in range(nL):
            s = LST[li]; dffl = dff
            n_shared = int(round(sf * B))
            if n_shared == 0:
                shared_idx = torch.tensor([], dtype=torch.long, device=dev)
            elif mode == "random":
                g = torch.Generator(device=dev).manual_seed(1000 * seed + li)
                shared_idx = torch.randperm(dffl, generator=g, device=dev)[:n_shared]
            else:  # freq
                shared_idx = s['freq'].topk(n_shared).indices
            shared_mask = torch.zeros(dffl, device=dev); shared_mask[shared_idx] = 1.0
            routed_pool = (shared_mask < 0.5).nonzero().flatten()
            route_budget = B - n_shared
            grp_local = kmeans(F.normalize(s['Wup'][routed_pool], dim=1), min(KROUTE, len(routed_pool)), seed=seed)
            Kc = int(grp_local.max().item()) + 1
            gsz = torch.zeros(Kc, device=dev)
            grp_full = torch.full((dffl,), 0, dtype=torch.long, device=dev)
            routed_is = torch.zeros(dffl, device=dev)
            rn = torch.zeros(s['a'].shape[0], Kc, device=dev)
            for g in range(Kc):
                ix = routed_pool[(grp_local == g).nonzero().flatten()]
                gsz[g] = len(ix); grp_full[ix] = g; routed_is[ix] = 1.0
                if len(ix):
                    rn[:, g] = (s['dev_a'][:, ix] @ s['Wd'][:, ix].T).norm(dim=1)
            router = mlp_fit(Z[li][tr], rn[tr], dev, seed=seed)
            inter = layers[li].intermediate
            inter._xbar = s['xbar'].float(); inter._P = s['P'].float(); inter._router = router
            inter._gsz = gsz.to(model.dtype); inter._route_budget = float(route_budget)
            inter._shared_mask = shared_mask.to(model.dtype).unsqueeze(0)
            inter._routed_grp_full = grp_full; inter._routed_is = routed_is.to(model.dtype)
        for li in range(nL):
            layers[li].intermediate.forward = types.MethodType(patched_intermediate, layers[li].intermediate)

    print(f"\n  MODEL-LEVEL MLM-CE, budget keep {BUDGET} (dense~2.04); 3 seeds:")
    print(f"  {'config':>16s} | {'seed0':>6s} | {'seed1':>6s} | {'seed2':>6s} | {'mean':>6s}")
    print("  " + "-" * 52)
    configs = [("route-all(G-MoE)", 0.0, "freq"),
               ("freq-shared-60", 0.6, "freq"),
               ("random-shared-60", 0.6, "random")]
    for tagc, sf, mode in configs:
        ces = []
        for sd in [0, 1, 2]:
            configure(sf, sd, mode); ces.append(mlm_ce())
        import numpy as np
        print(f"  {tagc:>16s} | {ces[0]:6.4f} | {ces[1]:6.4f} | {ces[2]:6.4f} | {np.mean(ces):6.4f}", flush=True)
    print("\nREAD: freq-shared < route-all across ALL seeds => robust win (not exp148 noise).")
    print("freq-shared < random-shared => the always-on SELECTION matters (not just individual handling).")


if __name__ == "__main__":
    main()
