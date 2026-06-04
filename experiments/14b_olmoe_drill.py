"""Experiment 14b — Drill into OlmoeExperts and OlmoeTopKRouter.

OlmoeExperts is fused (no iterable interface) and OlmoeTopKRouter
wraps a Linear differently than Switch. Find the actual W1/W2/W3
tensors and router weight so we can build the adapter.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


def main():
    from transformers import AutoModelForCausalLM

    print("Loading OLMoE (already cached)...")
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMoE-1B-7B-0924", dtype=torch.float16,
    )
    model.eval()

    mlp = model.model.layers[0].mlp
    print(f"\nmlp type: {type(mlp).__name__}")

    # Print all params of mlp
    print("\n--- All parameters of mlp ---")
    for n, p in mlp.named_parameters():
        print(f"  {n:60s}  shape {tuple(p.shape)}  dtype {p.dtype}")

    print("\n--- All buffers of mlp ---")
    for n, b in mlp.named_buffers():
        print(f"  {n:60s}  shape {tuple(b.shape)}  dtype {b.dtype}")

    # Inspect router
    print("\n--- Router (gate) detail ---")
    gate = mlp.gate
    for n, p in gate.named_parameters():
        print(f"  gate.{n}: shape {tuple(p.shape)}")
    print(f"  gate has 'weight'? {hasattr(gate, 'weight')}")
    if hasattr(gate, 'weight'):
        print(f"    weight shape: {tuple(gate.weight.shape)}")

    # Inspect experts
    print("\n--- Experts detail ---")
    experts = mlp.experts
    print(f"  experts type: {type(experts).__name__}")
    for n, p in experts.named_parameters():
        print(f"  experts.{n}: shape {tuple(p.shape)}")

    # Try a dummy forward to see signature
    print("\n--- Dummy forward through mlp ---")
    h = torch.randn(2, 4, 2048, dtype=torch.float16)
    try:
        out = mlp(h)
        if isinstance(out, tuple):
            print(f"  mlp(h) returned tuple of {len(out)}")
            for i, o in enumerate(out):
                if hasattr(o, 'shape'):
                    print(f"    [{i}] shape {tuple(o.shape)}")
                else:
                    print(f"    [{i}] type {type(o)}")
        else:
            print(f"  mlp(h) shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"  forward failed: {e}")


if __name__ == "__main__":
    main()
