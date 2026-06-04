"""Experiment 120 — Neuron MERGING (input-weight similarity) vs DROP on Phi-2 FFNs.

User's idea: merge neurons with similar weights. Mechanistically distinct from all the
drop-based methods we tried (which fail to cancellation): merging PRESERVES+SUMS
contributions instead of removing them, so it may sidestep cancellation. Analysis:
- INPUT-similar (w1_j ~ w1_k): activations match -> a_j w3_j + a_k w3_k ~ a*(w3_j+w3_k),
  so summing outputs MERGES losslessly AND cuts d_ff compute. This is the real candidate.
- OUTPUT-similar (w3_j ~ w3_k): (a_j+a_k)w3, but sum of two nonlinear activations isn't
  one neuron -> no clean merge, and both activations still computed -> no compute saving.
So we test input-similarity merging (Kim et al. 'Neuron Merging' style): cluster d_ff
units by w1 cosine into K' reps; member j folds into rep with scale s_j=(w1_j.w1_r)/||w1_r||^2
under homogeneity gelu(s z)~s gelu(z); merged W3 col = sum_j s_j w3_j. Compare to DROP
(keep K' highest-impact units, rest->mean) at the same K'. Headroom is bounded by how
near-duplicate the input directions are (diagnostic: NN-cosine, stable rank).

Phi-2 dense GeLU FFNs, calib from wikitext, FFN-output rel-L2. Run:
  python3 experiments/120_phi2_neuron_merge.py
"""
from __future__ import annotations

import sys
import pathlib
import statistics as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "microsoft/phi-2"
N_CALIB = 1024
N_EVAL = 512
COMP = [0.75, 0.5]                 # keep fraction of units (K'/d_ff)
LAYERS = [0, 6, 12, 18, 24, 30]    # subset of 32 layers for speed


def kmeans(X, k, iters=15):
    c = X[torch.randperm(X.shape[0], device=X.device)[:k]].clone()
    a = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a, c


def collect(model, ids, dev, layer, lo, hi):
    cap = []
    h = model.model.layers[layer].mlp.register_forward_pre_hook(
        lambda _m, a: cap.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    with torch.no_grad():
        model(ids[lo:hi].unsqueeze(0).to(dev))
    h.remove()
    return torch.cat(cap).float()


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    torch.set_grad_enabled(False)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0]

    print(f"Phi-2 neuron merge vs drop (input-similarity), layers {LAYERS}\n")
    print("  diagnostic: NN-cos = mean over units of max cosine to another unit (w1);")
    print("  stable-rank(W1) = ||W1||_F^2/||W1||_2^2 (high d_ff=10240 -> low=redundant)\n")
    res = {(c, m): [] for c in COMP for m in ("merge", "drop")}
    for li in LAYERS:
        mlp = model.model.layers[li].mlp
        w1 = mlp.fc1.weight.float()                  # [dff,d]
        w3 = mlp.fc2.weight.float()                  # [d,dff]
        dff = w1.shape[0]
        Hc = collect(model, ids, dev, li, 0, N_CALIB)
        He = collect(model, ids, dev, li, N_CALIB, N_CALIB + N_EVAL)
        a_e = mlp.activation_fn(He.half() @ w1.T.half()).float()    # [Te,dff]
        Ye = a_e @ w3.T
        yn = Ye.norm(dim=1)
        # diagnostic
        wn = w1 / (w1.norm(dim=1, keepdim=True) + 1e-8)
        sims = wn @ wn.T
        sims.fill_diagonal_(-1)
        nncos = sims.max(1).values.mean().item()
        srank = (w1.norm() ** 2 / torch.linalg.matrix_norm(w1, ord=2) ** 2).item()
        # mean activation for drop representative
        a_c = mlp.activation_fn(Hc.half() @ w1.T.half()).float()
        mean_a = a_c.mean(0)
        impact = (a_c.abs().mean(0) * w3.norm(dim=0))               # per-unit impact
        for c in COMP:
            K = max(1, int(c * dff))
            # ---- MERGE (input-similarity clustering) ----
            grp, _ = kmeans(wn, K)
            out = torch.zeros_like(Ye)
            for g in range(K):
                members = (grp == g).nonzero().flatten()
                if members.numel() == 0:
                    continue
                # representative = member with largest impact
                rep = members[impact[members].argmax()]
                wr = w1[rep]
                s = (w1[members] @ wr) / (wr @ wr + 1e-8)           # scale per member
                a_rep = mlp.activation_fn((He.half() @ wr.half())).float()  # [Te]
                w3merged = (s.unsqueeze(1) * w3[:, members].T).sum(0)       # [d]
                out += a_rep.unsqueeze(1) * w3merged.unsqueeze(0)
            res[(c, "merge")] += (((out - Ye).norm(dim=1)) / yn * 100).tolist()
            # ---- DROP (keep top-K by impact, rest -> mean) ----
            keep = impact.topk(K).indices
            m = torch.zeros(dff, device=dev)
            m[keep] = 1.0
            kept = a_e * m + mean_a.unsqueeze(0) * (1 - m)
            out_d = kept @ w3.T
            res[(c, "drop")] += (((out_d - Ye).norm(dim=1)) / yn * 100).tolist()
        print(f"  layer {li:2d}: NN-cos {nncos:.3f}  stable-rank {srank:6.1f}/{dff}",
              flush=True)

    print("\n  FFN-output rel-L2 error vs original (median %, lower=better):")
    print(f"  {'keep':>5s} | {'MERGE (input-sim)':>18s} | {'DROP (top-k+mean)':>18s}")
    print("  " + "-" * 50)
    for c in COMP:
        print(f"  {int(c*100):>4d}% | {st.median(res[(c,'merge')]):>18.1f} | "
              f"{st.median(res[(c,'drop')]):>18.1f}")
    print("\n  If MERGE << DROP -> input-similar neurons are redundant and merging")
    print("  (preserve+sum) beats removal, sidestepping cancellation -> compute win.")
    print("  If MERGE ~ DROP or large -> input directions are diverse (high stable rank);")
    print("  no near-duplicates to merge; merging gives no advantage over G-MoEfication.")


if __name__ == "__main__":
    main()
