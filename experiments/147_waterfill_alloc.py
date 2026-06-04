"""Experiment 146 — TASK-AWARE per-layer compression ALLOCATION (a 'beyond G-MoE' lever:
G-MoE compresses ~uniformly). If layers differ in task-sensitivity, allocate more compression
to robust layers, less to sensitive ones, at the SAME average keep -> better task at fixed budget.
mBERT, MoEfy all FFNs (fidelity ||r_g|| selection within layer). Measure MLM-CE (task):
 (A) per-layer sensitivity profile: compress ONLY layer li to keep 0.25, others full.
 (B) uniform(0.5) vs task-allocated(mean 0.5, keep ∝ sensitivity) -- static, then a quick finetune.
If allocated < uniform => task-aware allocation is a real lever beyond G-MoE's uniform.
Run: python3 experiments/146_taskaware_alloc.py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

NAME = "bert-base-multilingual-cased"
N_CALIB = 16384
N_EVAL = 8192
SEQ = 512
K = 64
AVG_KEEP = 0.5
MASK_P = 0.15
FT_STEPS = 400


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


def patched_intermediate(self, hidden_states):
    a = self.intermediate_act_fn(self.dense(hidden_states))
    m = self._mask
    return a * m + self._rep * (1 - m)


def main():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids_all = tok("\n\n".join(t for t in wt["text"] if t.strip()),
                  return_tensors="pt").input_ids[0]
    mask_id = tok.mask_token_id
    model = AutoModelForMaskedLM.from_pretrained(NAME).to(dev).eval()
    layers = model.bert.encoder.layer; nL = len(layers)

    def batches(start, n):
        return [ids_all[c0:c0 + SEQ].unsqueeze(0).to(dev) for c0 in range(start, start + n, SEQ)]

    @torch.no_grad()
    def mlm_ce(seed=0):
        gen = torch.Generator(device=dev).manual_seed(seed)
        tot = 0.0; ntok = 0
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
        hs.append(inter.register_forward_hook(
            (lambda li: (lambda _m, _i, o: capa[li].append(o.reshape(-1, o.shape[-1]).float())))(li)))
    for xb in batches(0, N_CALIB):
        model(xb)
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"mBERT: {nL} layers, dff={dff}", flush=True)

    for li in range(nL):
        inter = layers[li].intermediate
        Wup = inter.dense.weight.detach().float()
        Wd = layers[li].output.dense.weight.detach().float()
        a = torch.cat(capa[li]).to(dev); mean_a = a.mean(0); dev_a = a - mean_a
        grp = kmeans(F.normalize(Wup, dim=1).to(dev), K, seed=0)
        imp = torch.zeros(K, device=dev)
        for gi in range(K):
            ix = (grp == gi).nonzero().flatten()
            if ix.numel():
                imp[gi] = (dev_a[:, ix] @ Wd[:, ix].T.to(dev)).norm(dim=1).mean()
        inter._grp = grp; inter._imp = imp
        inter._rep0 = mean_a.to(model.dtype); inter._rep = inter._rep0
        inter.forward = types.MethodType(patched_intermediate, inter)
        del a, dev_a; gc.collect(); torch.cuda.empty_cache()

    def set_layer_keep(li, keep):
        inter = layers[li].intermediate
        kk = int(round(keep * K))
        gate = torch.zeros(K, device=dev)
        if kk > 0:
            gate[inter._imp.topk(kk).indices] = 1.0
        inter._mask = gate[inter._grp].to(model.dtype)
        inter._rep = inter._rep0

    def set_all(keep_vec):
        for li in range(nL):
            set_layer_keep(li, keep_vec[li])

    set_all([1.0] * nL)
    dense_ce = mlm_ce()
    print(f"\n  dense MLM-CE = {dense_ce:.4f}\n")

    # (A) per-layer sensitivity: compress only li to 0.25
    # (A) per-layer COST CURVE C_l(keep): compress ONLY li to each grid keep
    grid = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1]
    print(f"  building per-layer cost curves over keep grid {grid} ...", flush=True)
    cost = torch.zeros(nL, len(grid))   # CE when only layer li at grid[j]
    for li in range(nL):
        for j, kp in enumerate(grid):
            set_all([1.0] * nL)
            set_layer_keep(li, kp)
            cost[li, j] = mlm_ce() - dense_ce
        print(f"    layer {li:2d}: " + " ".join(f"{cost[li,j]:+.3f}" for j in range(len(grid))), flush=True)

    # (B) GREEDY water-fill: start all at keep=1 (level 0), repeatedly drop the layer-step with
    # smallest marginal ΔCE, until average keep == AVG_KEEP. All layers same width => unit budget.
    lvl = [0] * nL                       # current grid index per layer
    target_units = AVG_KEEP * nL         # sum of keep
    def cur_keep():
        return sum(grid[lvl[li]] for li in range(nL))
    while cur_keep() > target_units + 1e-6:
        best = None
        for li in range(nL):
            if lvl[li] + 1 < len(grid):
                dcost = (cost[li, lvl[li] + 1] - cost[li, lvl[li]]).item()
                dkeep = grid[lvl[li]] - grid[lvl[li] + 1]
                ratio = dcost / dkeep      # marginal task-cost per unit compute saved
                if best is None or ratio < best[0]:
                    best = (ratio, li)
        if best is None:
            break
        lvl[best[1]] += 1
    keep_alloc = [grid[lvl[li]] for li in range(nL)]

    set_all([AVG_KEEP] * nL); ce_uniform = mlm_ce()
    set_all(keep_alloc); ce_alloc = mlm_ce()
    print(f"\n  avg keep {AVG_KEEP} (alloc avg {sum(keep_alloc)/nL:.3f}):")
    print(f"    uniform CE       = {ce_uniform:.4f}")
    print(f"    waterfill-alloc  = {ce_alloc:.4f}    (dense {dense_ce:.4f})")
    print(f"    alloc keeps: {[round(k,2) for k in keep_alloc]}")
    print("\nREAD: waterfill < uniform => principled task-aware allocation beats uniform G-MoE.")
    print("~equal/worse => even marginal-cost-optimal allocation is inert (exp119 holds, task metric).")


if __name__ == "__main__":
    main()
