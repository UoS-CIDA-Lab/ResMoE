"""Experiment 115 — Model-level impact of our improved G-MoEfication: does the FFN-
fidelity ceiling bind at the TASK level, and is the self-supervised router enough?

All prior numbers were FFN-output rel-L2 (strict). G-MoEfication reports ~96% TASK
performance, suggesting the residual stream absorbs most FFN error. We now measure the
real thing: patch EVERY OLMoE expert's internal SwiGLU FFN with our G-MoEfied version
(G unit-groups, STATIC mean representative, keep fraction), inside the live forward, and
measure prediction (argmax) divergence vs the unmodified model on held-out text. Two
router modes: oracle-resid (true per-group residual = the ceiling) and ridge-resid (cheap
deployable, closed-form, self-supervised on unlabeled calib). Per-layer input-PCA feats.

If oracle model-level divergence is small -> the FFN-fidelity ceiling does NOT bind the
task; the structure we measured is absorbed. If oracle is small but ridge is large -> the
router IS the model-level bottleneck (motivates distillation-trained routing). If both
small -> our deployable G-MoEfication is already task-sufficient.

GPU, live OLMoE. Run: python3 experiments/115_model_level_gmoe.py
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
KEEP = [0.85, 0.5]
CALIB_LO, CALIB_HI = 0, N_CALIB


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
    elif module._gm_mode == "mlp":
        z = (x.float() - module._gm_xbar.float()) @ module._gm_P.float()
        score = module._gm_mlp(z).to(u.dtype)
    elif module._gm_mode == "distill":
        z = (x.float() - module._gm_xbar.float()) @ module._gm_P.float()
        score = z @ module._gm_Wr_d                                # [n,G] float, grad
    else:
        z = (x - module._gm_xbar) @ module._gm_P
        score = z @ module._gm_Wr
    order = score.detach().argsort(dim=1, descending=True)
    so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < budget
    selg = torch.zeros(n, Gn, dtype=torch.bool, device=u.device)
    selg.scatter_(1, order, keep_ord)
    m_hard = selg[:, grp].to(u.dtype)                              # [n,dff] keep mask
    if getattr(module, "_gm_train", False):                        # straight-through
        s = score.float()
        s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)
        m_soft = torch.sigmoid(s)[:, grp].to(u.dtype)
        m = m_hard + (m_soft - m_soft.detach())
    else:
        m = m_hard
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
    torch.set_grad_enabled(False)        # no training anywhere; avoid graph retention
    layers = model.model.layers
    nL = len(layers)

    # --- reference predictions (unmodified) ---
    @torch.no_grad()
    def preds():
        out = []
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            x = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            out.append(model(x).logits[0, :-1].float().argmax(-1).cpu())
        return out
    bp = preds()

    # --- collect per-layer MoE-block inputs on calib ---
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
    HL = {li: torch.cat(cap[li]) for li in range(nL)}              # [T,d] fp16

    # --- build per-expert G-MoEfication params ---
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
            if routed.numel() < 8:                                 # fallback: all calib
                routed = torch.arange(Hc.shape[0], device=dev)
            Xe = Hc[routed]
            u = (F.silu(Xe @ exp.gate_proj.weight.float().T)
                 * (Xe @ exp.up_proj.weight.float().T))            # [ne,dff]
            mean_u = u.mean(0)
            Wn = exp.gate_proj.weight.float()
            grp = kmeans(Wn / (Wn.norm(dim=1, keepdim=True) + 1e-8), G)
            idx = [(grp == g).nonzero().flatten() for g in range(G)]
            gsz = torch.tensor([len(ix) for ix in idx],
                               device=dev, dtype=torch.float16)
            dev_u = u - mean_u
            dwf = exp.down_proj.weight.float()
            gr = torch.stack([(dev_u[:, ix] @ dwf[:, ix].T).norm(dim=1)
                              for ix in idx], 1)                    # [ne,G]
            z = (Xe - xbar) @ P.float()
            Wr = torch.linalg.solve(z.T @ z + 1e-2 * torch.eye(RFEAT, device=dev),
                                    z.T @ gr)                       # [RFEAT,G]
            exp._gm_grp = grp.to(dev)
            exp._gm_idx = [ix.to(dev) for ix in idx]
            exp._gm_mean_u = mean_u.half()
            exp._gm_gsz = gsz
            exp._gm_P = P
            exp._gm_xbar = xbar.half()
            exp._gm_Wr = Wr.half()
            exp._gm_mlp = mlp_fit(z, gr)                            # FFN-target router
        print(f"  layer {li} done", flush=True)

    # --- patch all expert forwards ---
    import types
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
    print(f"  {'mode':>12s} {'keep':>5s} | {'pred-diff':>9s} | {'cert<=(95%)':>11s}")
    print("  " + "-" * 46)
    for mode in ("oracle", "mlp", "ridge"):
        for keep in KEEP:
            set_cfg(mode, keep)
            ps = preds()
            kbad = tot = 0
            for a, b in zip(bp, ps):
                kbad += int((a != b).sum().item())
                tot += a.numel()
            print(f"  {mode:>12s} {int(keep*100):>4d}% | {kbad/tot*100:>8.2f}% | "
                  f"{cp_upper(kbad, tot)*100:>10.2f}%", flush=True)

    # ---- distillation router: label-free, MODEL-OUTPUT target, end-to-end ----
    DTOK, DCHUNK, DSTEPS = 192, 64, 80
    for p in model.parameters():
        p.requires_grad_(False)
    set_cfg("oracle", 1.0)                       # keep=1 == unmodified -> teacher
    teacher = []
    for c0 in range(0, DTOK, DCHUNK):
        x = ids[c0:c0 + DCHUNK].unsqueeze(0).to(dev)
        teacher.append(model(x).logits[0, :-1].float().log_softmax(-1))

    print("\ntraining distillation routers (label-free, model-output KL)...", flush=True)
    for keep in KEEP:
        for li in range(nL):
            for e in range(layers[li].mlp.num_experts):
                exp = layers[li].mlp.experts[e]
                exp._gm_Wr_d = torch.nn.Parameter(exp._gm_Wr.float().clone())
                exp._gm_train = True
        dparams = [layers[li].mlp.experts[e]._gm_Wr_d
                   for li in range(nL) for e in range(layers[li].mlp.num_experts)]
        opt = torch.optim.Adam(dparams, lr=1e-2)
        set_cfg("distill", keep)
        with torch.enable_grad():
            for step in range(DSTEPS):
                opt.zero_grad()
                tl = 0.0
                for j, c0 in enumerate(range(0, DTOK, DCHUNK)):
                    x = ids[c0:c0 + DCHUNK].unsqueeze(0).to(dev)
                    logp = model(x).logits[0, :-1].float().log_softmax(-1)
                    loss = (teacher[j].exp() * (teacher[j] - logp)).sum(-1).mean()
                    loss.backward()
                    tl += loss.item()
                opt.step()
        for li in range(nL):
            for e in range(layers[li].mlp.num_experts):
                layers[li].mlp.experts[e]._gm_train = False
        set_cfg("distill", keep)
        ps = preds()
        kbad = tot = 0
        for a, b in zip(bp, ps):
            kbad += int((a != b).sum().item())
            tot += a.numel()
        print(f"  {'distill':>12s} {int(keep*100):>4d}% | {kbad/tot*100:>8.2f}% | "
              f"{cp_upper(kbad, tot)*100:>10.2f}%  (final train KL {tl:.4f})", flush=True)

    print("\noracle = ceiling (true per-group residual). ridge/mlp = self-supervised")
    print("FFN-residual-target routers. distill = label-free MODEL-OUTPUT-target router")
    print("(end-to-end KL to the unmodified model, straight-through selection). If distill")
    print("<< mlp/ridge and approaches oracle -> the right router objective is model-output,")
    print("reachable WITHOUT task labels; if distill ~ mlp -> selection is the hard limit.")


if __name__ == "__main__":
    main()
