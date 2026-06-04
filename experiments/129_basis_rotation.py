"""Experiment 129 — Does a BASIS ROTATION lower the per-token oracle drop floor?

Motivated by user pushback: G-MoE is greedy; is there room? Tests whether an
output-preserving orthogonal rotation Q of the hidden space (a->Qa, Wd->WdQ^T,
exact y) makes per-token neuron-dropping easier than the original neuron basis.

Clean theory: compute-PRESERVING orthogonal maps = permutations only, and those
leave oracle unchanged (exp 114). Floor-LOWERING rotations (PCA) break compute-
independence (a rotated coord mixes all neurons -> must compute all of a). So this
measures whether the FLOOR is basis-dependent (science), knowing the helpful
rotation is deployment-useless (no compute saving).

Cache has only 70 tokens -> PCA looks artificially perfect (rank<=69). Here we
harvest N_CALIB live-model layer-0 MLP inputs to get the HONEST rank/floor.
Run: python3 experiments/129_basis_rotation.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 2048
KEEPS = [0.25, 0.5, 0.85]


def oracle_drop_err(a, Wd, keep):
    abar = a.mean(0)
    dev = a - abar
    score = dev.abs() * Wd.norm(dim=0)
    k = int(round(keep * a.shape[1]))
    idx = score.argsort(dim=1, descending=True)
    mask = torch.zeros_like(a)
    mask.scatter_(1, idx[:, :k], 1.0)
    a_app = a * mask + abar * (1 - mask)
    y = a @ Wd.T
    yhat = a_app @ Wd.T
    return ((y - yhat).norm(dim=1) / (y.norm(dim=1) + 1e-6)).mean().item()


def effrank(M, thr=0.90):
    s = torch.linalg.svdvals(M - M.mean(0))
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    return int((e < thr).sum().item()) + 1


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0][:N_CALIB]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)
    mlp0 = model.model.layers[0].mlp
    cap = []
    h = mlp0.register_forward_pre_hook(lambda _m, a: cap.append(a[0].reshape(-1, a[0].shape[-1]).float().cpu()))
    model(ids.unsqueeze(0).to(dev))
    h.remove()
    H = torch.cat(cap)
    print(f"harvested H = {tuple(H.shape)} layer-0 MLP inputs", flush=True)

    Wg = mlp0.experts[0].gate_proj.weight.detach().float().cpu()
    dff = Wg.shape[0]
    N = mlp0.num_experts
    rng = torch.Generator().manual_seed(1)
    Qr, _ = torch.linalg.qr(torch.randn(dff, dff, generator=rng))

    tags = ["orig", "random", "pca"]
    res = {t: {k: [] for k in KEEPS} for t in tags}
    eranks = []
    for e in range(N):
        ex = mlp0.experts[e]
        wg = ex.gate_proj.weight.detach().float().cpu()
        wu = ex.up_proj.weight.detach().float().cpu()
        wd = ex.down_proj.weight.detach().float().cpu()
        a = F.silu(H @ wg.T) * (H @ wu.T)
        eranks.append(effrank(a))
        # pca rotation from THIS expert's activation principal axes
        _, _, Vt = torch.linalg.svd(a - a.mean(0), full_matrices=True)
        Qp = Vt  # [dff, dff]
        variants = {"orig": (a, wd),
                    "random": (a @ Qr.T, wd @ Qr.T),
                    "pca": (a @ Qp.T, wd @ Qp.T)}
        for t, (av, wdv) in variants.items():
            for k in KEEPS:
                res[t][k].append(oracle_drop_err(av, wdv, k))
    print(f"\nactivation effrank@90 (of {dff}): mean {np.mean(eranks):.0f} "
          f"[{int(np.min(eranks))}-{int(np.max(eranks))}]  (T={H.shape[0]} tokens)\n")
    print("Per-token ORACLE drop error (rel-L2 FFN output), mean over 64 experts")
    print("  rotation | " + " | ".join(f"keep{int(k*100):>2d}%" for k in KEEPS))
    print("  " + "-" * 40)
    for t in tags:
        print(f"  {t:8s} | " + " | ".join(f"{np.mean(res[t][k])*100:6.1f}%" for k in KEEPS), flush=True)
    print("\norig=neuron basis(G-MoE). random=fixed random Q. pca=activation principal axes.")
    print("permutation(compute-preserving) would equal orig exactly. pca lowers floor but needs")
    print("FULL a to compute any rotated coord => no compute saving (deployment-useless).")


if __name__ == "__main__":
    main()
