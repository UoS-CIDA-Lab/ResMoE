"""Experiment 124 — Does a DETERMINISTIC exact-gate router beat the LEARNED routers at
MODEL level, where exp 115 saw learned routers amplify through depth?

exp 123 (FFN level): gate-exact routing (score = per-group ||silu(gate)*||w3||||) ties the
learned mlp-pca and BOTH plateau far from oracle; gate-exact loses on the compute frontier
(must compute all gates to route). BUT exp 115 found the learned (mlp/ridge) routers'
small per-FFN gap AMPLIFIES through 16 layers (FFN-level mlp closes ~87% of the ridge->
oracle gap, but only ~20% at model level), partly an overfit/generalization failure (the
naive distill router even OVERFIT). A gate-exact router is DETERMINISTIC: it reads the
actual gate on every token, with NO calib->eval generalization gap. Hypothesis: even if it
ties the learned router per-FFN, it may amplify LESS through depth at model level.

Same harness as exp 115 (patch every OLMoE expert's SwiGLU FFN, STATIC mean rep, measure
argmax prediction divergence vs unmodified model on held-out text), routers:
  oracle : true per-group residual ||W_down[:,g] @ (u-mean)_g||         (ceiling, not deployable)
  ridge  : closed-form self-supervised FFN-residual-target router        (exp 115 deployable)
  mlp    : small MLP self-supervised FFN-residual-target router          (exp 115 deployable)
  gate   : DETERMINISTIC, per-group ||silu(gate_j)*||w_down_j|| ||       (ours, no learning)

Read: gate < mlp/ridge at model level => deterministic exact-gate routing IS more depth-
robust (a real, if compute-costly, model-level win). gate ~ mlp/ridge => FFN-level tie
holds through depth; routing wall confirmed end-to-end -> pivot to the certificate.

GPU, live OLMoE. Run: python3 experiments/124_gate_router_model_level.py
"""
from __future__ import annotations

import sys
import pathlib
import math
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 2048
N_EVAL = 512
CHUNK = 256
G = 64
RFEAT = 64
KEEP = [0.5, 0.667, 0.75, 0.85, 0.9]
CALIB_LO, CALIB_HI = 0, N_CALIB


def eff_compute(mode, keep):
    """fraction of dense-FFN compute (3 matmuls). learned router computes only kept
    units across gate+up+down -> 3k/3 = k. gate router must compute ALL gates to route,
    then up+down on kept -> (1+2k)/3. oracle is the (non-deployable) ceiling."""
    if mode in ("mlp", "ridge"):
        return keep
    if mode == "gate":
        return (1.0 + 2.0 * keep) / 3.0
    return float("nan")            # oracle: ceiling, not deployable


def cp_upper(k, n, conf=0.95):
    try:
        from scipy.stats import beta
        return 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))
    except Exception:
        return min(1.0, k / n + math.sqrt(math.log(1.0 / (1.0 - conf)) / (2 * n)))


def kmeans(X, k, iters=15):
    c = X[torch.randperm(X.shape[0], device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a


def mlp_fit(X, Y, hidden=64, steps=300, lr=5e-3, wd=1e-4):
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            F.mse_loss(net(X), Y).backward()
            opt.step()
    return net.eval()


def gm_forward(module, x):
    u = module.act_fn(module.gate_proj(x)) * module.up_proj(x)     # [n,dff]
    grp, mean_u, gsz = module._gm_grp, module._gm_mean_u, module._gm_gsz
    n, dff = u.shape
    Gn = gsz.shape[0]
    budget = module._gm_keep * dff
    if module._gm_mode == "oracle":
        dev = u - mean_u
        score = torch.empty(n, Gn, device=u.device, dtype=u.dtype)
        for g in range(Gn):
            ix = module._gm_idx[g]
            score[:, g] = (dev[:, ix] @ module.down_proj.weight[:, ix].T).norm(dim=1)
    elif module._gm_mode == "gate":                                # deterministic, no learning
        gsig = module.act_fn(module.gate_proj(x)).abs() * module._gm_w3n   # [n,dff]
        score = torch.empty(n, Gn, device=u.device, dtype=u.dtype)
        for g in range(Gn):
            ix = module._gm_idx[g]
            score[:, g] = gsig[:, ix].norm(dim=1)
    elif module._gm_mode == "mlp":
        z = (x.float() - module._gm_xbar.float()) @ module._gm_P.float()
        score = module._gm_mlp(z).to(u.dtype)
    else:                                                          # ridge
        z = (x - module._gm_xbar) @ module._gm_P
        score = z @ module._gm_Wr
    order = score.argsort(dim=1, descending=True)
    so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < budget
    selg = torch.zeros(n, Gn, dtype=torch.bool, device=u.device)
    selg.scatter_(1, order, keep_ord)
    m = selg[:, grp].to(u.dtype)                                   # [n,dff] keep mask
    return module.down_proj(u * m + mean_u.unsqueeze(0) * (1 - m))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)
    layers = model.model.layers
    nL = len(layers)

    @torch.no_grad()
    def preds():
        out = []
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            x = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            out.append(model(x).logits[0, :-1].float().argmax(-1).cpu())
        return out
    bp = preds()

    cap = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(layers[li].mlp.register_forward_pre_hook(
            (lambda li: (lambda _m, a: cap[li].append(
                a[0].detach().reshape(-1, a[0].shape[-1]))))(li)))
    with torch.no_grad():
        model(ids[CALIB_LO:CALIB_HI].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    HL = {li: torch.cat(cap[li]) for li in range(nL)}

    print(f"building G-MoEfication params: {nL} layers x "
          f"{layers[0].mlp.num_experts} experts, G={G} ...", flush=True)
    for li in range(nL):
        mlp = layers[li].mlp
        N, K = mlp.num_experts, mlp.top_k
        Hc = HL[li].float()
        xbar = Hc.mean(0)
        _, _, Vt = torch.linalg.svd(Hc - xbar, full_matrices=False)
        P = Vt[:RFEAT].T.half()
        logits = Hc @ mlp.gate.weight.float().T
        top = logits.topk(K, -1).indices
        for e in range(N):
            exp = mlp.experts[e]
            routed = (top == e).any(1).nonzero().flatten()
            if routed.numel() < 8:
                routed = torch.arange(Hc.shape[0], device=dev)
            Xe = Hc[routed]
            u = (F.silu(Xe @ exp.gate_proj.weight.float().T)
                 * (Xe @ exp.up_proj.weight.float().T))
            mean_u = u.mean(0)
            Wn = exp.gate_proj.weight.float()
            grp = kmeans(Wn / (Wn.norm(dim=1, keepdim=True) + 1e-8), G)
            idx = [(grp == g).nonzero().flatten() for g in range(G)]
            gsz = torch.tensor([len(ix) for ix in idx],
                               device=dev, dtype=torch.float16)
            dev_u = u - mean_u
            dwf = exp.down_proj.weight.float()
            gr = torch.stack([(dev_u[:, ix] @ dwf[:, ix].T).norm(dim=1)
                              for ix in idx], 1)
            z = (Xe - xbar) @ P.float()
            Wr = torch.linalg.solve(z.T @ z + 1e-2 * torch.eye(RFEAT, device=dev),
                                    z.T @ gr)
            exp._gm_grp = grp.to(dev)
            exp._gm_idx = [ix.to(dev) for ix in idx]
            exp._gm_mean_u = mean_u.half()
            exp._gm_gsz = gsz
            exp._gm_P = P
            exp._gm_xbar = xbar.half()
            exp._gm_Wr = Wr.half()
            exp._gm_w3n = dwf.norm(dim=0).half()                   # ||W_down[:,j]||
            exp._gm_mlp = mlp_fit(z, gr)
        print(f"  layer {li} done", flush=True)

    for li in range(nL):
        for e in range(layers[li].mlp.num_experts):
            exp = layers[li].mlp.experts[e]
            exp._gm_orig = exp.forward
            exp.forward = types.MethodType(gm_forward, exp)

    def set_cfg(mode, keep):
        for li in range(nL):
            for e in range(layers[li].mlp.num_experts):
                exp = layers[li].mlp.experts[e]
                exp._gm_mode, exp._gm_keep = mode, keep

    print(f"\nmodel-level prediction divergence vs unmodified OLMoE "
          f"(N_EVAL={N_EVAL} tokens), all FFNs G-MoEfied:\n")
    print(f"  {'mode':>8s} {'keep':>5s} {'eff':>6s} | {'pred-diff':>9s} | "
          f"{'cert<=(95%)':>11s}")
    print("  " + "-" * 52)
    res = {}
    for mode in ("oracle", "gate", "mlp", "ridge"):
        for keep in KEEP:
            set_cfg(mode, keep)
            ps = preds()
            kbad = tot = 0
            for a, b in zip(bp, ps):
                kbad += int((a != b).sum().item())
                tot += a.numel()
            res[(mode, keep)] = (kbad / tot * 100, eff_compute(mode, keep))
            ef = eff_compute(mode, keep)
            print(f"  {mode:>8s} {int(keep*100):>4d}% {ef:>6.3f} | "
                  f"{kbad/tot*100:>8.2f}% | {cp_upper(kbad, tot)*100:>10.2f}%", flush=True)

    # --- matched-compute frontier: interpolate each mode's pred-diff at a target eff ---
    def curve(mode):
        pts = sorted((eff_compute(mode, k), res[(mode, k)][0]) for k in KEEP)
        return pts

    def at_eff(mode, target):                # piecewise-linear interp on (eff,pred-diff)
        pts = curve(mode)
        if target <= pts[0][0]:
            return pts[0][1]
        if target >= pts[-1][0]:
            return pts[-1][1]
        for (e0, v0), (e1, v1) in zip(pts, pts[1:]):
            if e0 <= target <= e1:
                return v0 + (v1 - v0) * (target - e0) / (e1 - e0)
        return pts[-1][1]

    print("\nMATCHED-COMPUTE FRONTIER (the honest test): pred-diff at equal eff-compute.")
    print(f"  {'eff':>6s} | {'gate':>8s} | {'mlp':>8s} | {'ridge':>8s} | winner")
    print("  " + "-" * 52)
    for target in [0.55, 0.60, 0.667, 0.75, 0.85, 0.90]:
        g, m, r = at_eff("gate", target), at_eff("mlp", target), at_eff("ridge", target)
        win = min((g, "gate"), (m, "mlp"), (r, "ridge"))[1]
        print(f"  {target:>6.3f} | {g:>7.2f}% | {m:>7.2f}% | {r:>7.2f}% | {win}")
    print("\ngate wins at matched eff => exact-gate routing is a real frontier improvement")
    print("for MoE-preserving G-MoEfication (training-free, depth-robust). learned wins =>")
    print("the gate's full-matmul cost outweighs its accuracy; fixed-keep win is not a")
    print("per-FLOP win -> the honest deliverable is the certificate, not the method.")


if __name__ == "__main__":
    main()
