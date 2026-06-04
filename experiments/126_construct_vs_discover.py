"""Experiment 126 — Construction (trained) vs discovery (static): does relaxing the
no-training constraint rescue cheap MoE decomposition of a gated FFN?

exp 107-125: STATIC discovery is capped for gated FFNs (no latent structure to find).
Thesis: training is CONSTRUCTION -- it imposes modularity that wasn't there. We distill the
dense FFN output (MSE), at MATCHED active compute, and ask:
  (a) does training beat the STATIC neuron-drop baseline?  (trained vs init error)
  (b) does input-conditioned ROUTING add value over plain distillation?  (MoE vs small)

Each student is INITIALIZED as the width-h pruned real SwiGLU (keep top-h neurons by
||W_down[:,j]||*mean|silu(g_j)*u_j|), so init error == static neuron-drop and TRAINED error
measures gradient construction's gain. MoE: C experts init from that pruned FFN (+noise),
learned router trained with STRAIGHT-THROUGH top-1 (forward hard / backward soft, so the
HARD-routing objective is optimized directly -> train==eval) + Switch load-balance + weight
decay. Reported: init, trained-hard (deployable), trained-soft (routing-discretization-free
ceiling, MoE only).

Matched active compute: small eff=h/dff; MoE top-1 eff=(C+3h)/(3*dff) ~ h/dff (router tiny).
Metric = rel-L2 FFN-output err vs dense (static G-MoE neuron-drop ~38% @ eff .5, ~20% @
eff .85). OLMoE experts as dense SwiGLU. Run: python3 experiments/126_construct_vs_discover.py
"""
from __future__ import annotations

import sys
import pathlib
import statistics as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
LAYER = 0
N_TOK = 40000
N_EXPERTS = 4
CALIB_FRAC = 0.6
CONFIGS = [("small h256", 1, 256), ("MoE C8 h256", 8, 256),
           ("small h512", 1, 512), ("MoE C8 h512", 8, 512)]
STEPS = 4000
BATCH = 2048
LR = 5e-4
WD = 0.0


class SmallMoE(nn.Module):
    def __init__(self, d, h, C, init):
        super().__init__()
        self.C, self.h = C, h
        Wg, Wu, Wd = init
        nz = 0.0 if C == 1 else 0.02
        dv = Wg.device
        self.Wg = nn.Parameter(Wg.repeat(C, 1, 1) + nz * Wg.std() * torch.randn(C, h, d, device=dv))
        self.Wu = nn.Parameter(Wu.repeat(C, 1, 1) + nz * Wu.std() * torch.randn(C, h, d, device=dv))
        self.Wd = nn.Parameter(Wd.repeat(C, 1, 1) + nz * Wd.std() * torch.randn(C, d, h, device=dv))
        self.router = nn.Linear(d, C).to(dv) if C > 1 else None
        if self.router is not None:
            self.router.weight.data *= 0.01
            self.router.bias.data.zero_()

    def expert_out(self, x):
        g = F.silu(torch.einsum("nd,chd->nch", x, self.Wg)) \
            * torch.einsum("nd,chd->nch", x, self.Wu)
        return torch.einsum("nch,cdh->ncd", g, self.Wd)

    def forward(self, x, mode="soft"):
        if self.C == 1:
            g = F.silu(x @ self.Wg[0].T) * (x @ self.Wu[0].T)
            return g @ self.Wd[0].T, None
        E = self.expert_out(x)
        logits = self.router(x)
        soft = logits.softmax(1)
        if mode == "hard":
            idx = logits.argmax(1)
            return E[torch.arange(x.shape[0], device=x.device), idx], None
        if mode == "st":                                     # straight-through top-1
            idx = logits.argmax(1)
            hard = F.one_hot(idx, self.C).to(soft.dtype)
            w = hard + soft - soft.detach()
            aux = self.C * (hard.mean(0) * soft.mean(0)).sum()    # Switch load-balance
            return torch.einsum("nc,ncd->nd", w, E), aux
        return torch.einsum("nc,ncd->nd", soft, E), None     # pure soft


def prune_init(W1, W2, W3, h, Hc):
    g = F.silu(Hc @ W1.T) * (Hc @ W2.T)
    imp = g.abs().mean(0) * W3.norm(dim=0)
    keep = imp.topk(h).indices
    return W1[keep].contiguous(), W2[keep].contiguous(), W3[:, keep].contiguous()


def collect_inputs(model, ids, dev, layer, n, win=2048):
    cap = []
    h = model.model.layers[layer].mlp.register_forward_pre_hook(
        lambda _m, a: cap.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    got = 0
    with torch.no_grad():
        for s in range(0, ids.shape[0] - 1, win):
            model(ids[s:s + win].unsqueeze(0).to(dev))
            got += min(win, ids.shape[0] - s)
            if got >= n:
                break
    h.remove()
    return torch.cat(cap)[:n].float()


def rel(net, X, o, on, mode):
    with torch.no_grad():
        pred, _ = net(X, mode=mode)
    return ((pred - o).norm(dim=1) / on * 100).tolist()


def train_one(net, x_c, o_c, C):
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    on = o_c.norm(dim=1).clamp(min=1e-8)
    last = 0.0
    with torch.enable_grad():
        for _ in range(STEPS):
            bi = torch.randint(0, x_c.shape[0], (BATCH,), device=x_c.device)
            opt.zero_grad()
            pred, aux = net(x_c[bi], mode="st" if C > 1 else "soft")
            mse = F.mse_loss(pred, o_c[bi])
            loss = mse + (0.01 * aux if aux is not None else 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            last = ((pred.detach() - o_c[bi]).norm(dim=1)
                    / on[bi] * 100).median().item()
    return net.eval(), last


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    H = collect_inputs(model, ids, dev, LAYER, N_TOK)
    mlp = model.model.layers[LAYER].mlp
    d = H.shape[1]
    EW = {e: (mlp.experts[e].gate_proj.weight.detach().float(),
              mlp.experts[e].up_proj.weight.detach().float(),
              mlp.experts[e].down_proj.weight.detach().float())
          for e in range(N_EXPERTS)}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    T = H.shape[0]
    perm = torch.randperm(T, device=dev)
    nc = int(CALIB_FRAC * T)
    Hc, He = H[perm[:nc]], H[perm[nc:]]
    ne = He.shape[0]

    def eff(C, h):
        return (C * d + 3 * d * h) / (3 * d * 1024)

    print(f"construct vs discover: layer {LAYER}, {N_EXPERTS} experts as dense SwiGLU, "
          f"d={d}, dff=1024")
    print(f"calib={len(Hc)} eval={ne}; init=pruned real FFN; ST top-1 routing, "
          f"{STEPS} steps, wd={WD}\n")

    ia = {l: [] for l, _, _ in CONFIGS}
    ha = {l: [] for l, _, _ in CONFIGS}
    sa = {l: [] for l, C, _ in CONFIGS if C > 1}
    trl = {l: [] for l, _, _ in CONFIGS}

    for e in EW:
        W1e, W2e, W3e = EW[e]
        o_c = (F.silu(Hc @ W1e.T) * (Hc @ W2e.T)) @ W3e.T
        o_e = (F.silu(He @ W1e.T) * (He @ W2e.T)) @ W3e.T
        on = o_e.norm(dim=1).clamp(min=1e-8)
        for lab, C, h in CONFIGS:
            net = SmallMoE(d, h, C, prune_init(W1e, W2e, W3e, h, Hc)).to(dev)
            ia[lab] += rel(net, He, o_e, on, "hard" if C > 1 else "soft")
            net, trloss = train_one(net, Hc, o_c, C)
            trl[lab].append(trloss)
            ha[lab] += rel(net, He, o_e, on, "hard" if C > 1 else "soft")
            if C > 1:
                sa[lab] += rel(net, He, o_e, on, "soft")
        print(f"  expert {e} done", flush=True)

    print("\nrel-L2 FFN-output err vs dense (median %): init(static) -> TRAINED")
    print(f"  {'config':>14s} {'eff':>6s} | {'init':>7s} | {'trained':>8s} | "
          f"{'soft-ceil':>9s} | {'train-err':>9s} | gain")
    print("  " + "-" * 72)
    for lab, C, h in CONFIGS:
        i, t = st.median(ia[lab]), st.median(ha[lab])
        sv = f"{st.median(sa[lab]):>9.1f}" if C > 1 else f"{'-':>9s}"
        print(f"  {lab:>14s} {eff(C, h):>6.2f} | {i:>7.1f} | {t:>8.1f} | {sv} | "
              f"{st.mean(trl[lab]):>9.1f} | {i - t:>+5.1f}")
    print(f"\n  STATIC ref (same metric): G-MoE neuron-drop ~38% @ eff .5, ~20% @ eff .85.")
    print("\nREAD: trained << init => construction beats static. MoE-trained < small-trained")
    print("=> routing adds value once trained. soft-ceil << hard => discretization/overfit")
    print("is the bottleneck, not capacity. trained ~ init across the board => even")
    print("construction barely helps at this data budget.")


if __name__ == "__main__":
    main()
