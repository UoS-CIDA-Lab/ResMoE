"""Experiment 128 — Training-free router optimization: rank-r GATE-SKETCH routing.

exp 124 (model level): the EXACT-gate router (deterministic, training-free, score = per-group
||silu(gate_j)*||w_down_j||||) BEATS the learned mlp/ridge routers (depth-robust, no calib→
eval gap) at fixed keep, but LOSES on the per-FLOP frontier because it must compute the FULL
gate matmul to route (eff=(1+2k)/3). exp 123: gate stable rank ~45 (of 1024) — very low. exp
123's sketch FAILED, but it sketched gate AND up and reconstructed the tail-dominated activation
deviation; it never tried sketching ONLY the gate and routing by gate-magnitude.

IDEA (training-free router optimization): route by silu(rank-r GATE-sketch)*||w_down|| . Since
the gate is low-stable-rank (~45), a rank-~64 sketch ≈ exact gate, so this should inherit exp
124's exact-gate quality (> learned, depth-robust) at the SAME ~3% overhead as a learned router
(sketch cost r(d+dff) << full gate d*dff) -> finally a per-FLOP frontier WIN, with NO training.

Same exp-124 harness (patch every OLMoE FFN, argmax pred-divergence vs unmodified, STATIC rep).
Routers: oracle (ceiling) | gate (exact, expensive) | gatesk-{16,32,64,128} (cheap, ours) |
mlp, ridge (learned baselines). eff: mlp/ridge=k+PCA; gatesk-r=k+r(d+dff)/(3 d dff); gate=(1+2k)/3.
READ: gatesk-r matches gate-exact (saturates by r~64, gate sr~45) AND beats mlp at matched eff
=> training-free router optimization is a real per-FLOP win. gatesk ~ mlp => sketch loses the
signal; the exact-gate advantage needs the full gate. Run: python3 experiments/128_gate_sketch_router_model.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 2048
N_EVAL = 512
CHUNK = 256
G = 64
RFEAT = 64
RMAX = 128
KEEP = [0.5, 0.667, 0.75, 0.85, 0.9]
SKR = [16, 32, 64, 128]


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
    u = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
    grp, mean_u, gsz = module._gm_grp, module._gm_mean_u, module._gm_gsz
    n, dff = u.shape
    Gn = gsz.shape[0]
    budget = module._gm_keep * dff
    mode = module._gm_mode
    if mode == "oracle":
        dev = u - mean_u
        score = torch.empty(n, Gn, device=u.device, dtype=u.dtype)
        for g in range(Gn):
            ix = module._gm_idx[g]
            score[:, g] = (dev[:, ix] @ module.down_proj.weight[:, ix].T).norm(dim=1)
    elif mode == "gate" or mode.startswith("gatesk"):
        if mode == "gate":
            g_ = module.gate_proj(x)
        else:
            r = module._gm_skr
            g_ = ((x @ module._gm_gV[:r].T) * module._gm_gS[:r]) @ module._gm_gU[:, :r].T
        gsig = module.act_fn(g_).abs() * module._gm_w3n
        score = torch.empty(n, Gn, device=u.device, dtype=u.dtype)
        for g in range(Gn):
            ix = module._gm_idx[g]
            score[:, g] = gsig[:, ix].norm(dim=1)
    elif mode == "mlp":
        z = (x.float() - module._gm_xbar.float()) @ module._gm_P.float()
        score = module._gm_mlp(z).to(u.dtype)
    else:
        z = (x - module._gm_xbar) @ module._gm_P
        score = z @ module._gm_Wr
    order = score.argsort(dim=1, descending=True)
    so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < budget
    selg = torch.zeros(n, Gn, dtype=torch.bool, device=u.device)
    selg.scatter_(1, order, keep_ord)
    m = selg[:, grp].to(u.dtype)
    return module.down_proj(u * m + mean_u.unsqueeze(0) * (1 - m))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    import types
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
    d = model.config.hidden_size
    dff = layers[0].mlp.experts[0].gate_proj.weight.shape[0]

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
        model(ids[:N_CALIB].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    HL = {li: torch.cat(cap[li]) for li in range(nL)}

    print(f"building params: {nL}x{layers[0].mlp.num_experts} experts, G={G}, "
          f"d={d}, dff={dff} ...", flush=True)
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
            gpw = exp.gate_proj.weight.float()
            u = F.silu(Xe @ gpw.T) * (Xe @ exp.up_proj.weight.float().T)
            mean_u = u.mean(0)
            grp = kmeans(gpw / (gpw.norm(dim=1, keepdim=True) + 1e-8), G)
            idx = [(grp == g).nonzero().flatten() for g in range(G)]
            gsz = torch.tensor([len(ix) for ix in idx], device=dev, dtype=torch.float16)
            dwf = exp.down_proj.weight.float()
            dev_u = u - mean_u
            gr = torch.stack([(dev_u[:, ix] @ dwf[:, ix].T).norm(dim=1) for ix in idx], 1)
            z = (Xe - xbar) @ P.float()
            Wr = torch.linalg.solve(z.T @ z + 1e-2 * torch.eye(RFEAT, device=dev), z.T @ gr)
            U1, S1, V1 = torch.linalg.svd(gpw, full_matrices=False)
            exp._gm_grp = grp.to(dev)
            exp._gm_idx = [ix.to(dev) for ix in idx]
            exp._gm_mean_u = mean_u.half()
            exp._gm_gsz = gsz
            exp._gm_P, exp._gm_xbar, exp._gm_Wr = P, xbar.half(), Wr.half()
            exp._gm_w3n = dwf.norm(dim=0).half()
            exp._gm_gU = U1[:, :RMAX].half()
            exp._gm_gS = S1[:RMAX].half()
            exp._gm_gV = V1[:RMAX].half()
            exp._gm_mlp = mlp_fit(z, gr)
        print(f"  layer {li} done", flush=True)

    for li in range(nL):
        for e in range(layers[li].mlp.num_experts):
            exp = layers[li].mlp.experts[e]
            exp.forward = types.MethodType(gm_forward, exp)

    def set_cfg(mode, keep):
        skr = int(mode.split("-")[1]) if mode.startswith("gatesk") else 0
        for li in range(nL):
            for e in range(layers[li].mlp.num_experts):
                exp = layers[li].mlp.experts[e]
                exp._gm_mode, exp._gm_keep, exp._gm_skr = mode, keep, skr

    def eff(mode, keep):
        if mode in ("mlp", "ridge"):
            return keep + RFEAT * d / (3 * d * dff)
        if mode.startswith("gatesk"):
            r = int(mode.split("-")[1])
            return keep + r * (d + dff) / (3 * d * dff)
        if mode == "gate":
            return (1.0 + 2.0 * keep) / 3.0
        return float("nan")

    modes = ["oracle", "gate"] + [f"gatesk-{r}" for r in SKR] + ["mlp", "ridge"]
    print(f"\nmodel-level pred-divergence vs unmodified OLMoE (N_EVAL={N_EVAL}):\n")
    print(f"  {'mode':>10s} {'keep':>5s} {'eff':>6s} | {'pred-diff':>9s}")
    print("  " + "-" * 40)
    res = {}
    for mode in modes:
        for keep in KEEP:
            if mode == "oracle" and keep not in (KEEP[0], KEEP[-1]):
                continue
            set_cfg(mode, keep)
            ps = preds()
            kbad = tot = 0
            for a, b in zip(bp, ps):
                kbad += int((a != b).sum().item()); tot += a.numel()
            res[(mode, keep)] = (kbad / tot * 100, eff(mode, keep))
            print(f"  {mode:>10s} {int(keep*100):>4d}% {eff(mode, keep):>6.3f} | "
                  f"{kbad/tot*100:>8.2f}%", flush=True)

    def at_eff(mode, target):
        pts = sorted((res[(mode, k)][1], res[(mode, k)][0]) for k in KEEP
                     if (mode, k) in res)
        if target <= pts[0][0]:
            return pts[0][1]
        if target >= pts[-1][0]:
            return pts[-1][1]
        for (e0, v0), (e1, v1) in zip(pts, pts[1:]):
            if e0 <= target <= e1:
                return v0 + (v1 - v0) * (target - e0) / (e1 - e0)
        return pts[-1][1]

    print("\nMATCHED-eff frontier (pred-diff at equal eff-compute), training-free vs learned:")
    cmp = ["gatesk-64", "gate", "mlp", "ridge"]
    print(f"  {'eff':>6s} | " + " | ".join(f"{m:>9s}" for m in cmp) + " | winner")
    print("  " + "-" * 64)
    for tgt in [0.55, 0.60, 0.70, 0.80, 0.90]:
        vals = [(at_eff(m, tgt), m) for m in cmp]
        win = min(vals)[1]
        print(f"  {tgt:>6.3f} | " + " | ".join(f"{v:>8.2f}%" for v, _ in vals)
              + f" | {win}")
    print("\nREAD: gatesk-64 ~ gate (saturates, gate sr~45) AND gatesk-64 < mlp at matched eff")
    print("=> training-free router optimization (rank-r gate sketch) is a real per-FLOP win.")
    print("gatesk ~ mlp / >> gate => sketch loses the signal; exact-gate edge needs full gate.")


if __name__ == "__main__":
    main()
