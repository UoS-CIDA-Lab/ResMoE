"""Adapter for google/switch-base-8 → our certified-MoE framework.

Switch Transformers use top-1 routing and ReLU FFN, both natively
supported by our toy_moe code. This module:

  1. Loads the HF model.
  2. Extracts one MoE block (router + 8 ExpertFFN) into a structure
     that our existing cert_moe code can consume.
  3. Hooks into the model to collect calibration hidden states at the
     INPUT of that MoE block.

Switch-base-8 specifics:
  - d_model = 768, d_ff = 3072
  - 8 experts per MoE layer
  - Top-K = 1 (not 2)
  - ReLU activation
  - Router has bias=False
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# Standard MoE layer locations in Switch-base-8.
# Pattern: encoder/decoder block i, layer 1 (FFN layer), .mlp is sparse MoE
# Only some blocks have MoE — Switch alternates dense and sparse.
SWITCH_MOE_PATHS = [
    f"encoder.block.{i}.layer.1.mlp" for i in (1, 3, 5, 7, 9, 11)
] + [
    f"decoder.block.{i}.layer.2.mlp" for i in (1, 3, 5, 7, 9, 11)
]


@dataclass
class SwitchMoEConfig:
    """Mirror of ToyMoEConfig but reflecting real Switch-base-8."""
    d_model: int = 768
    d_ff: int = 3072
    n_experts: int = 8
    top_k: int = 1
    n_clone_pairs: int = 0


class SwitchExpertWrapper(nn.Module):
    """Wrap a SwitchTransformersDenseActDense as ExpertFFN-compatible.

    Switch's expert is: y = wo(relu(wi(x)))
    Our cert_moe code expects: forward(h) → h, with linear W1 (no bias)
    and W2 (no bias), composed with ReLU.

    Switch's wi has bias=False by default. wo also.

    We cast all weights to float32 to match the router (which Switch
    keeps in float32 by design — `router_dtype: float32` in the config).
    Mixed-precision dtype mismatches break our bound code.
    """

    def __init__(self, switch_expert: nn.Module):
        super().__init__()
        # Cast to float32 to match router; keep as Parameter so we can
        # use .detach() etc. uniformly.
        self.W1 = nn.Parameter(switch_expert.wi.weight.float())
        self.W2 = nn.Parameter(switch_expert.wo.weight.float())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(h @ self.W1.T) @ self.W2.T


class SwitchMoEWrapper(nn.Module):
    """Make a Switch MoE block look like our ToyMoE for downstream code."""

    def __init__(
        self, switch_mlp: nn.Module, cfg: SwitchMoEConfig,
    ):
        super().__init__()
        self.cfg = cfg
        # Switch router: classifier linear w/ no bias
        # `switch_mlp.router.classifier` is a nn.Linear(d_model, n_experts)
        self.router = switch_mlp.router.classifier
        # Switch experts: ModuleDict of SwitchTransformersDenseActDense
        self.experts = nn.ModuleList([
            SwitchExpertWrapper(switch_mlp.experts[f"expert_{i}"])
            for i in range(cfg.n_experts)
        ])
        self.clone_pairs = []

    def route(self, h: torch.Tensor) -> dict:
        logits = self.router(h)
        topk_vals, topk_idx = logits.topk(self.cfg.top_k, dim=-1)
        topk_probs = torch.softmax(topk_vals, dim=-1)
        return {
            "logits": logits,
            "topk_idx": topk_idx,
            "topk_probs": topk_probs,
        }

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, d = h.shape
        info = self.route(h)
        out = torch.zeros_like(h)
        for b in range(B):
            for slot in range(self.cfg.top_k):
                i = info["topk_idx"][b, slot].item()
                g = info["topk_probs"][b, slot]
                out[b] = out[b] + g * self.experts[i](h[b:b+1]).squeeze(0)
        return out, info


def load_switch_block(
    path: str = "encoder.block.1.layer.1.mlp",
    model_name: str = "google/switch-base-8",
) -> tuple[SwitchMoEWrapper, object, object]:
    """Load Switch-base-8 and return a wrapped MoE block + model + tokenizer.

    Returns (wrapper, model, tokenizer).
    """
    from transformers import AutoTokenizer, AutoModel

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)

    # Drill down to the MoE block
    obj = model
    for piece in path.split("."):
        if piece.isdigit():
            obj = obj[int(piece)]
        else:
            obj = getattr(obj, piece)

    cfg = SwitchMoEConfig()
    wrapper = SwitchMoEWrapper(obj, cfg)
    return wrapper, model, tok


def collect_hidden_states(
    model, tokenizer, path: str,
    texts: list[str], batch_size: int = 8,
) -> torch.Tensor:
    """Capture the hidden states arriving at the MoE block via forward hook.

    Returns a tensor of shape [total_tokens, d_model].
    """
    # Find the parent of the MoE block; hook on input to mlp
    obj = model
    for piece in path.split("."):
        if piece.isdigit():
            obj = obj[int(piece)]
        else:
            obj = getattr(obj, piece)

    captured = []

    def hook(_module, inputs, _output):
        # inputs[0]: [B, L, d_model]
        captured.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]))

    handle = obj.register_forward_hook(hook)

    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=128,
                )
                # Switch is encoder-decoder; we need decoder_input_ids for
                # the model forward. For encoder MoE blocks we only need
                # to drive the encoder.
                _ = model.encoder(**{k: v for k, v in enc.items()
                                     if k in ("input_ids", "attention_mask")})
    finally:
        handle.remove()

    return torch.cat(captured, dim=0)
