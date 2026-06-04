"""Experiment 15b — One-time OLMoE extraction + cache.

Load OLMoE once, extract router + experts for layer 0, collect
calibration hidden states, and save everything to a .pt file. Then
delete the model and exit cleanly. Subsequent experiments load only
the cache (~200MB) instead of the full 14GB model.

Run: python experiments/15b_olmoe_cache.py
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gc

import torch

from cert_moe.olmoe_adapter import (
    load_olmoe_block, collect_hidden_states_olmoe,
)


SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models predict outputs from inputs.",
    "Neural networks consist of layers of neurons.",
    "Mixture of experts increases model capacity.",
    "Climate change is a pressing global concern.",
    "Education shapes future generations.",
    "Technology drives social change.",
    "Mathematics describes the universe.",
]


def main():
    cache_path = (
        pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
    )
    if cache_path.exists():
        print(f"Cache already exists: {cache_path}")
        print("Delete it manually if you want to rebuild.")
        return

    print("Loading OLMoE layer 0...")
    moe, model, tok = load_olmoe_block(layer_idx=0)

    print("\nCollecting calibration hidden states...")
    H = collect_hidden_states_olmoe(
        model, tok, layer_idx=0, texts=SAMPLES, batch_size=2,
    ).to(moe.router.weight.dtype)
    print(f"  H shape {tuple(H.shape)}")

    # Extract everything we need into plain tensors
    router_w = moe.router.weight.detach().clone()
    experts_W1 = torch.stack([e.W1.detach().clone() for e in moe.experts])
    experts_W2 = torch.stack([e.W2.detach().clone() for e in moe.experts])
    experts_W3 = torch.stack([e.W3.detach().clone() for e in moe.experts])

    # Drop everything
    del model, moe
    gc.collect()

    cache = {
        "router_weight": router_w,
        "experts_W1": experts_W1,
        "experts_W2": experts_W2,
        "experts_W3": experts_W3,
        "H": H,
        "d_model": 2048,
        "d_ff": 1024,
        "n_experts": 64,
        "top_k": 8,
        "norm_topk_prob": False,
    }
    torch.save(cache, cache_path)
    print(f"\nSaved cache to {cache_path}")
    print(f"  router    : {tuple(router_w.shape)}")
    print(f"  experts_W1: {tuple(experts_W1.shape)}")
    print(f"  experts_W2: {tuple(experts_W2.shape)}")
    print(f"  experts_W3: {tuple(experts_W3.shape)}")
    print(f"  H         : {tuple(H.shape)}")
    import os
    print(f"  size      : "
          f"{os.path.getsize(cache_path)/1024**2:.1f} MB")


if __name__ == "__main__":
    main()
