"""Experiment 18 — Full-model WikiText perplexity after merging OLMoE layer 0.

Bridges the gap flagged in PAPER_OUTLINE §10.1: rel_div measures one MoE
layer's output divergence, NOT downstream model quality. Here we merge
layer 0's experts (64 → N') with cosine vs cert-aware plans, swap the
merged modules back into the *real* OLMoE-1B-7B, and measure the actual
WikiText-2 perplexity change.

Merge plans are built from the cached layer-0 hidden states
(olmoe_layer0_cache.pt). The merged single layer is installed into the
live model and scored; the original modules are restored between runs.

Run: python3 experiments/18_olmoe_perplexity.py
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
CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


class LiteMoE:
    """Minimal object exposing the .cfg/.router that the merge utilities
    and check_conditions need (they never touch .experts)."""
    def __init__(self, router_weight, n_experts, top_k, d_model, d_ff):
        self.cfg = ToySwiGLUMoEConfig(
            d_model=d_model, d_ff=d_ff, n_experts=n_experts,
            top_k=top_k, n_clone_pairs=0,
        )
        self.router = nn.Linear(d_model, n_experts, bias=False)
        with torch.no_grad():
            self.router.weight.copy_(router_weight)


def cosine_distance(block):
    """1 - cosine over concatenated (gate_proj, up_proj, down_proj)."""
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
    """Build (gate, experts, N') for the merged plan, matching OlmoeMLP."""
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
    """Sliding-window perplexity over a 1-D tensor of token ids."""
    nll_sum, n_tok, n_win = 0.0, 0, 0
    for begin in range(0, ids.shape[0] - 1, stride):
        end = min(begin + ctx, ids.shape[0])
        inp = ids[begin:end].unsqueeze(0).to(device)
        target = inp.clone()
        target[:, :-1] = inp[:, 1:]
        target[:, -1] = -100
        out = model(inp)
        logits = out.logits.float()
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = target[:, :-1].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        valid = (shift_labels != -100).sum().item()
        nll_sum += loss.item()
        n_tok += valid
        n_win += 1
        if max_windows and n_win >= max_windows:
            break
    return math.exp(nll_sum / max(n_tok, 1)), n_tok


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model.to(device).eval()

    block = model.model.layers[LAYER_IDX].mlp
    K_orig = block.top_k
    N_orig = block.num_experts
    d_model = block.gate.weight.shape[1]
    d_ff = block.experts[0].gate_proj.weight.shape[0]
    mlp_config = block.experts[0].config
    print(f"Layer {LAYER_IDX} MoE: N={N_orig}, K={K_orig}, "
          f"d={d_model}, d_ff={d_ff}")

    # Keep originals to restore between runs
    orig_gate = block.gate
    orig_experts = block.experts

    # Calibration hidden states + plans (CPU, fp32)
    cache = torch.load(CACHE, weights_only=True, map_location="cpu")
    H = cache["H"].float()
    del cache
    router_w = block.gate.weight.detach().float().cpu()
    freq = routing_freq(H, router_w, K_orig, N_orig)
    print(f"Calibration H={tuple(H.shape)}, "
          f"usage min {freq.min():.0f}/max {freq.max():.0f}")
    d_cos = cosine_distance(block).cpu()

    lite = LiteMoE(router_w, N_orig, K_orig, d_model, d_ff)

    # WikiText-2 test tokens
    print("Loading WikiText-2 test...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    # Cap for runtime: ~40 windows of 1024 ≈ 40k tokens
    print(f"Total test tokens: {ids.shape[0]}")
    MAX_WINDOWS = 40

    print("\nBaseline (original) perplexity...")
    ppl_base, ntok = compute_ppl(model, ids, device, max_windows=MAX_WINDOWS)
    print(f"  baseline PPL = {ppl_base:.4f}  ({ntok} tokens)")

    results = {"baseline": ppl_base}
    dtype = orig_gate.weight.dtype

    for target_n in (32,):
        print(f"\n{'='*64}\n  Merge layer {LAYER_IDX}: 64 → {target_n}\n{'='*64}")

        # --- cosine plan ---
        eps = find_cutoff(d_cos, target_n)
        plan_cos = complete_linkage_clusters(d_cos, epsilon=eps)
        # --- cert-aware plan ---
        max_b = d_cos[d_cos.isfinite()].max().item() * 1.001
        plan_ca = cert_aware_greedy_merge(
            lite, H, d_cos, target_n_clusters=target_n,
            max_bound=max_b, delta_em=0.1, bound_weight=0.0,
        )

        for tag, plan in (("cosine", plan_cos), ("cert-aware", plan_ca)):
            m = check_conditions(H, router_w, plan, K=K_orig, delta_threshold=0.1)
            g, ex, Np = build_merged_modules(
                block, plan, freq, mlp_config, device, dtype)
            block.gate, block.experts = g, ex
            block.num_experts, block.top_k = Np, min(K_orig, Np)
            ppl, _ = compute_ppl(model, ids, device, max_windows=MAX_WINDOWS)
            # restore
            block.gate, block.experts = orig_gate, orig_experts
            block.num_experts, block.top_k = N_orig, K_orig
            cert = m.all_certified.float().mean().item()
            dppl = (ppl - ppl_base) / ppl_base * 100
            results[f"{tag}_{target_n}"] = ppl
            print(f"  {tag:11s} → PPL {ppl:.4f}  (Δ {dppl:+.2f}%), "
                  f"cert {cert*100:.1f}%")

    print(f"\n{'='*64}\n  SUMMARY (WikiText-2, layer {LAYER_IDX} merged only)\n{'='*64}")
    print(f"  baseline           PPL {results['baseline']:.4f}")
    for tn in (32,):
        for tag in ("cosine", "cert-aware"):
            k = f"{tag}_{tn}"
            d = (results[k] - results['baseline']) / results['baseline'] * 100
            print(f"  {tag:11s} 64→{tn}  PPL {results[k]:.4f}  (Δ {d:+.2f}%)")


if __name__ == "__main__":
    main()
