"""Experiment 221 — RANK diagnostic: is a near-lossless K=128 grouping even POSSIBLE on SwiGLU, or is
the group→neuron gap a fundamental high-rank limit? (decisive, no heuristic search.) A balanced K-group
partition can match per-neuron selection only if the per-token KEEP patterns live near a ~K-dim subspace
(K clusters can then align the active sets). Measure: top-K spectral ENERGY fraction of
  - keep-indicator B [N,dff] (binary top-B membership) — what fraction of co-keep variance in top-K dims
  - contribution matrix C [N,dff] = (a-abar)*vn (continuous)
and effective (stable) rank. If top-128 energy ~1 / stable-rank ~128 => low-rank => a good K=128 grouping
EXISTS (keep searching). If top-128 energy is small / stable-rank >> 128 => high-rank => NO fixed K=128
grouping can align per-token active sets => the deployable structured floor is FUNDAMENTAL.
Qwen(SwiGLU) keep50/25, averaged over layers. Run: HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/221_rank_diagnostic.py
"""
from __future__ import annotations
import sys, pathlib, gc, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
N_CALIB = 8192
CHUNK = 512
KS = [128, 256, 512]
KEEPS = [0.50, 0.25]


def keep_topB_neuron(score, B):
    thr = score.kthvalue(score.shape[1] - B + 1, dim=1, keepdim=True).values
    return score >= thr


def topk_energy(M, ks):
    """fraction of squared-singular-value energy in top-k, for each k in ks; + stable rank."""
    M = M - M.mean(0, keepdim=True)                  # center columns (neurons) over tokens
    total = (M * M).sum().item()
    q = min(max(ks) + 16, M.shape[0] - 1, M.shape[1] - 1)
    _, S, _ = torch.svd_lowrank(M, q=q)
    s2 = (S * S)
    fracs = {k: (s2[:k].sum().item() / total) for k in ks}
    stable = total / (S[0].item() ** 2)              # ||M||_F^2 / sigma_max^2 (lower bound on rank)
    return fracs, stable


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    code = []
    for sp in ["train", "test", "validation", "prompt"]:
        try:
            code += load_dataset("mbpp", split=sp, trust_remote_code=True)["code"]
        except Exception:
            pass
    ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        def parts(l): return l.mlp.down_proj
        arch = "SwiGLU"
    else:
        layers = model.transformer.h
        def parts(l): return l.mlp.c_proj
        arch = "GeLU"
    nL = len(layers); downs = [parts(layers[li]) for li in range(nL)]
    print(f"  MODEL={MODEL} [{arch}]  layers={nL}", flush=True)
    torch.set_grad_enabled(False)

    capa = {li: [] for li in range(nL)}
    hs = [downs[li].register_forward_pre_hook(
        (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li))
        for li in range(nL)]
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"  dff={dff}  (a near-lossless K-grouping needs per-token keep-patterns in a ~K-dim subspace)\n", flush=True)

    # accumulate over layers
    agg = {('B', bf): {k: [] for k in KS} for bf in KEEPS}
    aggC = {k: [] for k in KS}
    stB = {bf: [] for bf in KEEPS}; stC = []
    for li in range(nL):
        a = torch.cat(capa[li]).float().to(dev)
        Wd = downs[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        C = (a - abar) * vn                                  # contribution matrix
        fc, sc = topk_energy(C, KS)
        for k in KS:
            aggC[k].append(fc[k])
        stC.append(sc)
        for bf in KEEPS:
            B = int(round(bf * dff))
            Bind = keep_topB_neuron((a - abar).abs() * vn, B).float()
            fb, sb = topk_energy(Bind, KS)
            for k in KS:
                agg[('B', bf)][k].append(fb[k])
            stB[bf].append(sb)
        capa[li] = None
        del a, Wd, C; gc.collect(); torch.cuda.empty_cache()

    def mean(x):
        return sum(x) / len(x)

    print(f"  === top-K spectral ENERGY fraction (mean over {nL} layers) ===", flush=True)
    print(f"  contribution matrix C:   " + "  ".join(f"top{k}={mean(aggC[k]):.3f}" for k in KS)
          + f"  | stable-rank≈{mean(stC):.0f}", flush=True)
    for bf in KEEPS:
        print(f"  keep-indicator @keep{int(bf*100)}: " + "  ".join(f"top{k}={mean(agg[('B',bf)][k]):.3f}" for k in KS)
              + f"  | stable-rank≈{mean(stB[bf]):.0f}", flush=True)
    print(f"\nREAD: if top128 energy ~1 and stable-rank ~128 => low-rank => a near-lossless K=128 grouping", flush=True)
    print("EXISTS (our search just hasn't found it). If top128 energy small and stable-rank >> 128 (toward", flush=True)
    print("dff) => per-token keep-patterns are HIGH-RANK => no fixed K=128 grouping can align them =>", flush=True)
    print("the deployable structured-selection floor on SwiGLU is FUNDAMENTAL (not a search failure).", flush=True)


if __name__ == "__main__":
    main()
