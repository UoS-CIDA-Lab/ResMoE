"""Experiment 06 — Inspect Switch-base-8 architecture.

Goal: understand the model so we can extract MoE layers cleanly.

Switch Transformers [Fedus+22] use top-1 routing — simpler than our toy
(which used top-2). Our framework should handle both K values.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


def main():
    from transformers import AutoTokenizer, AutoModel

    print("Loading google/switch-base-8 (this will download ~600MB on first run)...")
    tok = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = AutoModel.from_pretrained("google/switch-base-8")
    model.eval()

    # Top-level structure
    print("\n--- Top-level modules ---")
    for name, _ in model.named_children():
        print(f"  {name}")

    # T5 has encoder + decoder; Switch MoE is in encoder/decoder blocks
    # Find MoE layers
    print("\n--- Searching for MoE / router / expert modules ---")
    moe_layers = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "MoE" in cls or "Router" in cls or "Expert" in cls or "Switch" in cls:
            moe_layers.append((name, cls))
    for name, cls in moe_layers[:20]:
        print(f"  {name:60s} | {cls}")
    if len(moe_layers) > 20:
        print(f"  ... and {len(moe_layers)-20} more")

    # Pick first MoE block, inspect deeply
    print("\n--- First MoE block detail ---")
    first_moe = None
    for name, module in model.named_modules():
        if "experts" in name and isinstance(module, torch.nn.ModuleDict):
            first_moe = (name, module)
            break
    if first_moe is None:
        # Try harder
        for name, module in model.named_modules():
            cls = type(module).__name__
            if "SparseMLP" in cls or "MoE" in cls:
                first_moe = (name, module)
                break
    if first_moe:
        name, module = first_moe
        print(f"  Path: {name}")
        print(f"  Type: {type(module).__name__}")
        for cn, _ in module.named_children():
            print(f"    child: {cn}")

    # Config inspection
    print("\n--- Model config (MoE-relevant fields) ---")
    cfg = model.config
    for key in dir(cfg):
        if any(k in key.lower() for k in ("expert", "moe", "router", "d_model", "d_ff", "num_lay", "num_experts")):
            if not key.startswith("_"):
                try:
                    val = getattr(cfg, key)
                    if not callable(val):
                        print(f"  {key}: {val}")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
