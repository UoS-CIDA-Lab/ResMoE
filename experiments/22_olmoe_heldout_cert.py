"""Experiment 22 — Is the OLMoE cert% collapse a calibration-SIZE artifact?

Experiment 20 found OLMoE cert-aware cert% collapsed 28.6% (70 calib
tokens) → 0.1% (real eval). Experiment 21 found Switch cert% generalizes
fine (88.5%→82.8%) — but Switch used 1248 calib tokens for 8 experts/top-1,
while OLMoE used only 70 tokens for 64 experts/top-8 (severe undersampling).

This disentangles the confound: collect a LARGE token set from the live
OLMoE model, split into disjoint calibration / held-out, build the
cert-aware plan (64→32) on calibration, and measure cert% on BOTH.

  - held-out cert% ≈ calibration cert%  → 70 tokens was the problem;
    method generalizes on OLMoE too (only the quality issue remains).
  - held-out cert% still collapses       → the 64-expert/top-8 regime
    genuinely fails to generalize cert%.

Needs HF_HUB_DISABLE_XET=1. Run:
  HF_HUB_DISABLE_XET=1 python3 experiments/22_olmoe_heldout_cert.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cert_moe.toy_swiglu_moe import ToySwiGLUMoEConfig
from cert_moe.merge import cert_aware_greedy_merge, complete_linkage_clusters
from cert_moe.theorem1_conditions import check_conditions

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
LAYER_IDX = 0
TARGET_N = 32
N_CALIB = 1024
N_HELD = 1024


class LiteMoE:
    def __init__(self, router_weight, n_experts, top_k, d_model, d_ff):
        self.cfg = ToySwiGLUMoEConfig(
            d_model=d_model, d_ff=d_ff, n_experts=n_experts,
            top_k=top_k, n_clone_pairs=0,
        )
        self.router = nn.Linear(d_model, n_experts, bias=False)
        with torch.no_grad():
            self.router.weight.copy_(router_weight)


def cosine_distance(block):
    vecs = torch.stack([
        torch.cat([e.gate_proj.weight.flatten(),
                   e.up_proj.weight.flatten(),
                   e.down_proj.weight.flatten()]).float()
        for e in block.experts
    ])
    vecs = F.normalize(vecs, dim=-1)
    dist = 1.0 - vecs @ vecs.T
    dist.fill_diagonal_(0.0)
    return dist


def routing_freq(H, router_weight, K, n_experts):
    logits = H @ router_weight.T
    _, idx = logits.topk(K, dim=-1)
    freq = torch.zeros(n_experts)
    for k in range(K):
        freq.scatter_add_(0, idx[:, k], torch.ones(H.shape[0]))
    return freq


@torch.no_grad()
def collect_hidden(model, block, ids, device, target_tokens, ctx=512):
    captured = []
    total = 0

    def pre_hook(_m, args):
        captured.append(args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu())

    handle = block.register_forward_pre_hook(pre_hook)
    try:
        for begin in range(0, ids.shape[0] - 1, ctx):
            inp = ids[begin:begin + ctx].unsqueeze(0).to(device)
            model(inp)
            total = sum(c.shape[0] for c in captured)
            if total >= target_tokens:
                break
    finally:
        handle.remove()
    return torch.cat(captured)[:target_tokens]


def cert_breakdown(H, router_w, plan, K):
    m = check_conditions(H, router_w, plan, K=K, delta_threshold=0.1)
    return {
        "mc": m.mc.float().mean().item(),
        "ts": m.ts.float().mean().item(),
        "em": m.em.float().mean().item(),
        "cert": m.all_certified.float().mean().item(),
    }


def find_cutoff(dist, target_n):
    import math
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model.to(device).eval()

    block = model.model.layers[LAYER_IDX].mlp
    K, N = block.top_k, block.num_experts
    d_model = block.gate.weight.shape[1]
    d_ff = block.experts[0].gate_proj.weight.shape[0]
    router_w = block.gate.weight.detach().float().cpu()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]

    print(f"Collecting {N_CALIB + N_HELD} layer-{LAYER_IDX} hidden states...")
    H_all = collect_hidden(model, block, ids, device, N_CALIB + N_HELD)
    H_cal = H_all[:N_CALIB]
    H_held = H_all[N_CALIB:N_CALIB + N_HELD]
    print(f"  calibration: {H_cal.shape[0]}, held-out: {H_held.shape[0]}")

    freq = routing_freq(H_cal, router_w, K, N)
    d_cos = cosine_distance(block).cpu()
    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001
    lite = LiteMoE(router_w, N, K, d_model, d_ff)

    plans = {}
    eps = find_cutoff(d_cos, TARGET_N)
    plans["cosine"] = complete_linkage_clusters(d_cos, epsilon=eps)
    print(f"\nBuilding cert-aware plan on {N_CALIB} calib tokens (bw=0)...",
          flush=True)
    plans["cert-aware"] = cert_aware_greedy_merge(
        lite, H_cal, d_cos, target_n_clusters=TARGET_N,
        max_bound=max_b, delta_em=0.1, bound_weight=0.0,
    )

    print(f"\n{'='*70}")
    print(f"  OLMoE layer {LAYER_IDX}, 64→{TARGET_N}, "
          f"calib={N_CALIB} (vs exp20's 70)")
    print(f"{'='*70}")
    print(f"  {'method':12s} {'set':12s} "
          f"{'MC%':>6s} {'TS%':>6s} {'EM%':>6s} {'cert%':>7s}")
    print("  " + "-" * 52)
    summ = {}
    for name, plan in plans.items():
        for set_name, H in (("calibration", H_cal), ("held-out", H_held)):
            r = cert_breakdown(H, router_w, plan, K)
            summ[(name, set_name)] = r["cert"]
            print(f"  {name:12s} {set_name:12s} "
                  f"{r['mc']*100:>5.1f}% {r['ts']*100:>5.1f}% "
                  f"{r['em']*100:>5.1f}% {r['cert']*100:>6.1f}%")

    print(f"\n{'='*70}")
    print("  VERDICT — OLMoE cert% calibration → held-out (large calib)")
    print(f"{'='*70}")
    for name in plans:
        c = summ[(name, "calibration")] * 100
        h = summ[(name, "held-out")] * 100
        drop = "—" if c == 0 else f"{(c-h)/c*100:+.0f}% rel."
        print(f"  {name:12s}: {c:5.1f}%  →  {h:5.1f}%   ({drop})")
    print("\n  exp20 (70 calib): 28.6% → 0.1% (collapsed).")
    print("  If now robust → collapse was a calibration-SIZE artifact.")


if __name__ == "__main__":
    main()
