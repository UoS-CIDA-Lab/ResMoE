"""Experiment 19 — bound_weight sweep: quality vs certified-coverage trade-off.

Experiment 18 showed that pure cert-aware merging (bound_weight=0) wrecks
WikiText perplexity (+217% on OLMoE layer 0, 64→32) because it merges
functionally-INDEPENDENT (dissimilar) experts to satisfy the Theorem-1
conditions, and averaging dissimilar experts ruins the centroid.

The open question: is that quality collapse INHERENT to cert-aware, or an
artifact of the extreme bound_weight=0 setting? cert_aware_greedy_merge
scores each candidate merge as

    score = cert_fraction - bound_weight * normalized_distance

so raising bound_weight penalizes dissimilar merges, pulling the method
toward cosine-like (similar-expert) merging. This sweep traces the
perplexity-vs-cert% curve over bound_weight to find whether a middle
ground recovers quality while keeping non-trivial certified coverage.

Same setup as exp 18: real OLMoE-1B-7B, merge layer 0 only, 64→32,
WikiText-2 perplexity. Needs HF_HUB_DISABLE_XET=1 to download the model.

Run: HF_HUB_DISABLE_XET=1 python3 experiments/19_olmoe_boundweight_sweep.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cert_moe.toy_swiglu_moe import ToySwiGLUMoEConfig
from cert_moe.merge import (
    complete_linkage_clusters, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
LAYER_IDX = 0
TARGET_N = 32
SWEEP = [0.0, 0.5, 1.0, 2.0]
MAX_WINDOWS = 40
CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


class LiteMoE:
    """Minimal object exposing .cfg/.router for the merge utilities."""
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


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def build_merged_modules(block, plan, freq, mlp_config, device, dtype):
    from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
    d = block.gate.weight.shape[1]
    Np = len(plan.clusters)
    new_gate = nn.Linear(d, Np, bias=False).to(device=device, dtype=dtype)
    new_experts = nn.ModuleList()
    with torch.no_grad():
        for c, members in enumerate(plan.clusters):
            w = freq[members].float()
            w = w / w.sum() if w.sum() > 0 else torch.ones_like(w) / len(w)
            e = OlmoeMLP(mlp_config).to(device=device, dtype=dtype)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                acc = sum(
                    w[i] * getattr(block.experts[m], proj).weight.float()
                    for i, m in enumerate(members)
                )
                getattr(e, proj).weight.copy_(acc.to(dtype))
            new_gate.weight[c].copy_(
                torch.logsumexp(block.gate.weight[members].float(), dim=0).to(dtype)
            )
            new_experts.append(e)
    return new_gate, new_experts, Np


@torch.no_grad()
def compute_ppl(model, ids, device, ctx=1024, stride=1024, max_windows=None):
    nll_sum, n_tok, n_win = 0.0, 0, 0
    for begin in range(0, ids.shape[0] - 1, stride):
        end = min(begin + ctx, ids.shape[0])
        inp = ids[begin:end].unsqueeze(0).to(device)
        target = inp.clone()
        target[:, :-1] = inp[:, 1:]
        target[:, -1] = -100
        logits = model(inp).logits.float()
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = target[:, :-1].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        nll_sum += loss.item()
        n_tok += (shift_labels != -100).sum().item()
        n_win += 1
        if max_windows and n_win >= max_windows:
            break
    return math.exp(nll_sum / max(n_tok, 1))


def evaluate_plan(model, block, plan, freq, mlp_config, device, dtype,
                  ids, orig_gate, orig_experts, N_orig, K_orig):
    g, ex, Np = build_merged_modules(block, plan, freq, mlp_config, device, dtype)
    block.gate, block.experts = g, ex
    block.num_experts, block.top_k = Np, min(K_orig, Np)
    ppl = compute_ppl(model, ids, device, max_windows=MAX_WINDOWS)
    block.gate, block.experts = orig_gate, orig_experts
    block.num_experts, block.top_k = N_orig, K_orig
    return ppl, Np


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model.to(device).eval()

    block = model.model.layers[LAYER_IDX].mlp
    K_orig, N_orig = block.top_k, block.num_experts
    d_model = block.gate.weight.shape[1]
    d_ff = block.experts[0].gate_proj.weight.shape[0]
    mlp_config = block.experts[0].config
    dtype = block.gate.weight.dtype
    orig_gate, orig_experts = block.gate, block.experts

    cache = torch.load(CACHE, weights_only=True, map_location="cpu")
    H = cache["H"].float()
    del cache
    router_w = block.gate.weight.detach().float().cpu()
    freq = routing_freq(H, router_w, K_orig, N_orig)
    d_cos = cosine_distance(block).cpu()
    lite = LiteMoE(router_w, N_orig, K_orig, d_model, d_ff)
    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001

    print("Loading WikiText-2 test...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]

    print("Baseline perplexity...")
    ppl_base = compute_ppl(model, ids, device, max_windows=MAX_WINDOWS)
    print(f"  baseline PPL = {ppl_base:.4f}")

    rows = []  # (label, ppl, cert, n)

    # cosine anchor
    eps = find_cutoff(d_cos, TARGET_N)
    plan_cos = complete_linkage_clusters(d_cos, epsilon=eps)
    cert_cos = check_conditions(H, router_w, plan_cos, K=K_orig,
                                delta_threshold=0.1).all_certified.float().mean().item()
    ppl_cos, n_cos = evaluate_plan(model, block, plan_cos, freq, mlp_config,
                                   device, dtype, ids, orig_gate, orig_experts,
                                   N_orig, K_orig)
    rows.append(("cosine", ppl_cos, cert_cos, n_cos))
    print(f"\ncosine          → PPL {ppl_cos:.4f} "
          f"(Δ {(ppl_cos-ppl_base)/ppl_base*100:+.1f}%), cert {cert_cos*100:.1f}%")

    # bound_weight sweep
    for bw in SWEEP:
        print(f"\n[bound_weight={bw}] building cert-aware plan...", flush=True)
        plan = cert_aware_greedy_merge(
            lite, H, d_cos, target_n_clusters=TARGET_N,
            max_bound=max_b, delta_em=0.1, bound_weight=bw,
        )
        cert = check_conditions(H, router_w, plan, K=K_orig,
                                delta_threshold=0.1).all_certified.float().mean().item()
        ppl, Np = evaluate_plan(model, block, plan, freq, mlp_config, device,
                                dtype, ids, orig_gate, orig_experts,
                                N_orig, K_orig)
        rows.append((f"cert bw={bw}", ppl, cert, Np))
        print(f"cert bw={bw:<4} → PPL {ppl:.4f} "
              f"(Δ {(ppl-ppl_base)/ppl_base*100:+.1f}%), cert {cert*100:.1f}%")

    print(f"\n{'='*66}")
    print(f"  SUMMARY — OLMoE layer {LAYER_IDX}, 64→{TARGET_N}, WikiText-2")
    print(f"  baseline PPL {ppl_base:.4f}")
    print(f"{'='*66}")
    print(f"  {'method':14s} {'PPL':>9s} {'ΔPPL':>9s} {'cert%':>7s} {'n':>4s}")
    print("  " + "-" * 50)
    for label, ppl, cert, n in rows:
        d = (ppl - ppl_base) / ppl_base * 100
        print(f"  {label:14s} {ppl:>9.3f} {d:>+8.1f}% {cert*100:>6.1f}% {n:>4d}")


if __name__ == "__main__":
    main()
