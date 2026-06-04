"""Adapter for allenai/OLMoE-1B-7B-0924 → our certified-MoE framework.

OLMoE specifics:
  - 16 transformer layers, 64 experts per layer, top-K = 8
  - d_model = 2048, d_ff (per expert) = 1024
  - SwiGLU activation (silu of gate * up)
  - Fused expert weights:
       gate_up_proj: [n_experts, d_model, 2*d_ff]   (gate ⊕ up)
       down_proj   : [n_experts, d_ff, d_model]
  - norm_topk_prob = False — top-K probabilities NOT renormalized

The fused forward uses `aten::_grouped_mm` which is unavailable on CPU.
We work around this by extracting expert weights into our own
SwiGLUExpert modules (per-expert) and replacing the MLP's forward with
a plain Python loop for hidden-state collection. This is slow but works
on CPU; we only need short runs for calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cert_moe.toy_swiglu_moe import SwiGLUExpert


@dataclass
class OLMoEMoEConfig:
    d_model: int = 2048
    d_ff: int = 1024
    n_experts: int = 64
    top_k: int = 8
    n_clone_pairs: int = 0
    norm_topk_prob: bool = False   # OLMoE quirk


class OLMoEMoEWrapper(nn.Module):
    """Drop-in replacement that exposes OLMoE MoE block in our schema."""

    def __init__(self, olmoe_mlp: nn.Module, cfg: OLMoEMoEConfig):
        super().__init__()
        self.cfg = cfg
        # Router
        # OlmoeTopKRouter has `.weight` of shape [n_experts, d_model]
        # Build as nn.Linear without bias for consistency
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        with torch.no_grad():
            self.router.weight.copy_(olmoe_mlp.gate.weight.float())

        # Experts — extract per-expert weights from fused tensors
        gate_up = olmoe_mlp.experts.gate_up_proj.detach().float()
        down = olmoe_mlp.experts.down_proj.detach().float()
        # gate_up: [n_experts, d_model, 2*d_ff]  (used as h @ GU[i])
        # down:    [n_experts, d_model, d_ff]    (used as intermediate @ D[i].T)
        d_ff = cfg.d_ff
        self.experts = nn.ModuleList()
        for i in range(cfg.n_experts):
            e = SwiGLUExpert(cfg.d_model, cfg.d_ff)
            with torch.no_grad():
                # OLMoE forward: gate = h @ GU[i, :, :d_ff]
                # Our forward:   gate = h @ W1.T   (W1 shape [d_ff, d_model])
                # ⇒ W1 = GU[i, :, :d_ff].T
                e.W1.copy_(gate_up[i, :, :d_ff].T.contiguous())
                e.W2.copy_(gate_up[i, :, d_ff:].T.contiguous())
                # OLMoE: out = intermediate @ D[i].T  (D[i] shape [d_model, d_ff])
                # Our:   out = intermediate @ W3.T    (W3 shape [d_model, d_ff])
                # ⇒ W3 = D[i]  (no extra transpose)
                e.W3.copy_(down[i])
            self.experts.append(e)
        self.clone_pairs = []

    def route(self, h: torch.Tensor) -> dict:
        logits = self.router(h)
        topk_vals, topk_idx = logits.topk(self.cfg.top_k, dim=-1)
        if self.cfg.norm_topk_prob:
            topk_probs = F.softmax(topk_vals, dim=-1)
        else:
            # OLMoE uses raw softmax over full N then gathers top-K
            full_probs = F.softmax(logits, dim=-1)
            topk_probs = full_probs.gather(-1, topk_idx)
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


def _patch_olmoe_experts_forward(model):
    """Replace fused grouped_mm forward in OlmoeExperts with a slow Python
    loop, so the model can run on CPU. We need this only for collecting
    calibration hidden states via hooks.
    """
    # Find OlmoeExperts modules
    cls_name = "OlmoeExperts"

    def new_forward(self, hidden_states, selected_experts, routing_weights):
        # hidden_states: [B, L, d_model]  flattened by caller usually
        # selected_experts: [B*L, K]
        # routing_weights: [B*L, K]
        # OLMoE expert: gate_up = h @ gate_up_proj[i]
        #               gate, up = chunk(2)
        #               intermediate = silu(gate) * up
        #               out_i = intermediate @ down_proj[i]
        # We loop over selected (token, expert) pairs and accumulate.
        # Reshape: collapse batch and seq dims for token-wise routing.
        orig_shape = hidden_states.shape
        h = hidden_states.reshape(-1, orig_shape[-1]).float()
        out = torch.zeros_like(h)
        n_tokens, K = selected_experts.shape
        for t in range(n_tokens):
            for k in range(K):
                e = selected_experts[t, k].item()
                w = routing_weights[t, k]
                # Forward through expert e
                gu = h[t] @ self.gate_up_proj[e].float()
                g, u = gu.chunk(2, dim=-1)
                inter = torch.nn.functional.silu(g) * u
                # down_proj[e] is [d_model, d_ff]; need .T for inter @ ...
                out[t] = out[t] + w * (inter @ self.down_proj[e].float().T)
        return out.to(hidden_states.dtype).reshape(orig_shape)

    n_patched = 0
    for module in model.modules():
        if type(module).__name__ == cls_name:
            module.forward = new_forward.__get__(module, type(module))
            n_patched += 1
    return n_patched


def load_olmoe_block(layer_idx: int = 0, model_name: str = "allenai/OLMoE-1B-7B-0924"):
    """Load OLMoE and return (wrapper, model, tokenizer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)

    block = model.model.layers[layer_idx].mlp
    cfg = OLMoEMoEConfig()
    wrapper = OLMoEMoEWrapper(block, cfg)
    return wrapper, model, tok


def collect_hidden_states_olmoe(
    model, tokenizer, layer_idx: int,
    texts: list[str], batch_size: int = 2,
) -> torch.Tensor:
    """Capture hidden states arriving at layer_idx's MoE block.

    We patch the fused OlmoeExperts forward to a slow Python loop so
    the model can run on CPU.
    """
    n_patched = _patch_olmoe_experts_forward(model)
    print(f"  Patched {n_patched} OlmoeExperts.forward for CPU compatibility.")

    target = model.model.layers[layer_idx].mlp
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]))

    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=64,
                )
                _ = model(**{k: v for k, v in enc.items()
                             if k in ("input_ids", "attention_mask")})
    finally:
        handle.remove()

    return torch.cat(captured, dim=0)
