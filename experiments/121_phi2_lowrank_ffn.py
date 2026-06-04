"""Experiment 121 — Low-rank FACTORIZATION of Phi-2 FFNs vs DROP, at matched compute.

exp 120's diagnostic found W1 has surprisingly LOW stable rank (~17-103 of 10240): its
energy concentrates in a few singular directions. That is NOT neuron duplication (merge
failed) but SPECTRAL redundancy -> a different, data-flagged lever: factorize fc1 (and
fc2) as low-rank (W ~ A B, rank r), cutting compute to ~r(d+d_ff)/(d*d_ff) of the FFN.
The open question (vs our cancellation prior): does gelu(low-rank pre-activation) preserve
the function, or does the truncated spectral tail carry the input-dependent computation?

We SVD-truncate W1 and W3 to rank r (data-free, the directly-motivated test) and also an
activation-aware variant (preserve W1 over the calib input distribution). Compare FFN-
output rel-L2 to DROP (G-MoEfication: keep top-impact units, rest->mean) at the SAME
compute (keep_equiv = r(d+d_ff)/(d*d_ff)). Phi-2 dense GeLU FFNs, wikitext calib.

Run: python3 experiments/121_phi2_lowrank_ffn.py
"""
from __future__ import annotations

import sys
import pathlib
import statistics as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MODEL = "microsoft/phi-2"
N_CALIB = 1024
N_EVAL = 512
RANKS = [512, 1024, 1536]          # ~ keep_equiv 0.25 / 0.5 / 0.75
LAYERS = [0, 6, 12, 18, 24, 30]


def svd_trunc(W, r):
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vt[:r]


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
    act = model.model.layers[0].mlp.activation_fn

    d = model.config.hidden_size
    dff = model.config.intermediate_size
    keq = {r: r * (d + dff) / (d * dff) for r in RANKS}
    print(f"Phi-2 low-rank FFN vs drop; d={d} dff={dff}; "
          f"ranks {RANKS} ~ keep {[round(keq[r],3) for r in RANKS]}\n")

    res = {(r, m): [] for r in RANKS for m in ("lowrank-df", "lowrank-aa", "drop")}
    for li in LAYERS:
        mlp = model.model.layers[li].mlp
        W1 = mlp.fc1.weight.float()
        W3 = mlp.fc2.weight.float()
        Hc = collect(model, ids, dev, li, 0, N_CALIB)
        He = collect(model, ids, dev, li, N_CALIB, N_CALIB + N_EVAL)
        a_e = act(He.half() @ W1.T.half()).float()
        Ye = a_e @ W3.T
        yn = Ye.norm(dim=1)
        a_c = act(Hc.half() @ W1.T.half()).float()
        mean_a = a_c.mean(0)
        impact = a_c.abs().mean(0) * W3.norm(dim=0)
        # activation-aware basis for fc1: top input-covariance directions
        # min ||(W1-W1') Hc^T|| -> project W1 onto row-space spanned by top PCA of Hc
        Uc, Sc, Vtc = torch.linalg.svd(Hc - Hc.mean(0), full_matrices=False)
        for r in RANKS:
            # ---- data-free low-rank (SVD of weights) ----
            W1r = svd_trunc(W1, r)
            W3r = svd_trunc(W3, r)
            out = act((He @ W1r.T)) @ W3r.T
            res[(r, "lowrank-df")] += ((out - Ye).norm(dim=1) / yn * 100).tolist()
            # ---- activation-aware fc1 (preserve W1 over input dist), df fc3 ----
            Pr = Vtc[:r].T                                   # [d,r] top input dirs
            W1aa = (W1 @ Pr) @ Pr.T                           # rank-r in input space
            out2 = act((He @ W1aa.T)) @ W3r.T
            res[(r, "lowrank-aa")] += ((out2 - Ye).norm(dim=1) / yn * 100).tolist()
            # ---- drop at matched compute ----
            K = max(1, int(keq[r] * dff))
            keep = impact.topk(K).indices
            m = torch.zeros(dff, device=dev)
            m[keep] = 1.0
            kept = a_e * m + mean_a.unsqueeze(0) * (1 - m)
            res[(r, "drop")] += ((kept @ W3.T - Ye).norm(dim=1) / yn * 100).tolist()
        print(f"  layer {li} done", flush=True)

    print("\n  FFN-output rel-L2 vs original (median %, lower=better):")
    print(f"  {'rank':>5s} {'~keep':>6s} | {'lowrank-df':>11s} | "
          f"{'lowrank-aa':>11s} | {'drop':>8s}")
    print("  " + "-" * 52)
    for r in RANKS:
        print(f"  {r:>5d} {keq[r]:>5.2f}  | {st.median(res[(r,'lowrank-df')]):>11.1f} | "
              f"{st.median(res[(r,'lowrank-aa')]):>11.1f} | "
              f"{st.median(res[(r,'drop')]):>8.1f}")
    print("\n  lowrank << drop -> spectral factorization beats removal (the low stable-rank")
    print("  is functionally exploitable; a real compute lever). lowrank ~ drop or large ->")
    print("  the truncated tail carries the computation (cancellation again); SVD energy")
    print("  concentration does NOT imply functional low-rank.")


if __name__ == "__main__":
    main()
