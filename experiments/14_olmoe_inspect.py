"""Experiment 14 — Inspect OLMoE-1B-7B architecture.

OLMoE is a decoder-only SwiGLU MoE from Allen AI:
  - 16 transformer layers
  - 64 experts per layer
  - Top-K = 8
  - d_model = 2048
  - SwiGLU activation
  - RoPE positional encoding

Goal: confirm we can load it, identify MoE block structure, and prepare
for hidden-state collection.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading allenai/OLMoE-1B-7B-0924 (~14GB download on first run)...")
    print("Loading in fp16 to halve memory.")
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMoE-1B-7B-0924",
        dtype=torch.float16,
    )
    model.eval()

    print("\n--- Top-level modules ---")
    for name, _ in model.named_children():
        print(f"  {name}")

    print("\n--- Config (MoE-relevant) ---")
    cfg = model.config
    for key in ("hidden_size", "intermediate_size", "num_experts",
                "num_experts_per_tok", "num_hidden_layers",
                "moe_intermediate_size", "num_routed_experts",
                "shared_expert_intermediate_size", "norm_topk_prob",
                "hidden_act", "router_aux_loss_coef"):
        if hasattr(cfg, key):
            print(f"  {key}: {getattr(cfg, key)}")

    # Find the MoE block path
    print("\n--- Searching for MoE modules ---")
    moe_paths = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        if any(k in cls for k in ("MoE", "Expert", "Sparse", "Router")):
            moe_paths.append((name, cls))
    print(f"  Found {len(moe_paths)} candidates. First 10:")
    for n, c in moe_paths[:10]:
        print(f"    {n}  |  {c}")

    # Look at first decoder layer's mlp
    print("\n--- First layer mlp detail ---")
    first_mlp = model.model.layers[0].mlp
    print(f"  Type: {type(first_mlp).__name__}")
    for cn, _ in first_mlp.named_children():
        sub = getattr(first_mlp, cn)
        print(f"    {cn}: {type(sub).__name__}")

    # Find router and experts
    print("\n--- Router and expert structure ---")
    if hasattr(first_mlp, 'gate'):
        print(f"  gate: {first_mlp.gate}")
    if hasattr(first_mlp, 'experts'):
        print(f"  experts: {type(first_mlp.experts).__name__}")
        if hasattr(first_mlp.experts, '__iter__'):
            try:
                first_expert = first_mlp.experts[0]
                print(f"  expert[0] type: {type(first_expert).__name__}")
                for ecn, _ in first_expert.named_children():
                    esub = getattr(first_expert, ecn)
                    if hasattr(esub, 'weight'):
                        print(f"    expert.{ecn}: "
                              f"weight {tuple(esub.weight.shape)}")
                    else:
                        print(f"    expert.{ecn}: {type(esub).__name__}")
            except Exception as e:
                print(f"    inspection failed: {e}")


if __name__ == "__main__":
    main()
