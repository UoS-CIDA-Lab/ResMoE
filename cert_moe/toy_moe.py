"""Synthetic MoE layer for prototype testing.

Design choices:
- ReLU FFN (not SwiGLU) — bound derivation is trivial, lets us
  validate the framework end-to-end before tackling SwiGLU relaxation.
- Deliberately constructed "clone pairs" so we have ground-truth
  mergeable pairs to validate against.
- Top-K routing with softmax + log-sum-exp normalization, matching
  the formulation in our framework.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyMoEConfig:
    d_model: int = 64
    d_ff: int = 128
    n_experts: int = 16
    top_k: int = 2
    n_clone_pairs: int = 2     # number of deliberately-cloned (i,j) pairs
    clone_noise_std: float = 0.01  # how "nearly equal" the clones are


class ExpertFFN(nn.Module):
    """y = W2 @ ReLU(W1 @ h). No biases for cleaner bound derivation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(h @ self.W1.T) @ self.W2.T


class ToyMoE(nn.Module):
    """Top-K MoE layer with optional planted clone pairs.

    Forward returns (output, routing_info) where routing_info contains
    everything we need for downstream verification (logits, top-K, gating).
    """

    def __init__(self, cfg: ToyMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertFFN(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts)]
        )
        self.clone_pairs = self._plant_clones()

    def _plant_clones(self) -> list[tuple[int, int]]:
        """Make `n_clone_pairs` pairs of experts nearly identical."""
        pairs: list[tuple[int, int]] = []
        n = self.cfg.n_experts
        # Pair up the last 2k experts: (n-2k, n-2k+1), (n-2k+2, n-2k+3), ...
        k = self.cfg.n_clone_pairs
        with torch.no_grad():
            for p in range(k):
                i = n - 2 * (k - p)
                j = i + 1
                pairs.append((i, j))
                # Copy expert i to j, then add small noise
                noise_std = self.cfg.clone_noise_std
                self.experts[j].W1.copy_(
                    self.experts[i].W1 + noise_std * torch.randn_like(self.experts[i].W1)
                )
                self.experts[j].W2.copy_(
                    self.experts[i].W2 + noise_std * torch.randn_like(self.experts[i].W2)
                )
        return pairs

    def route(self, h: torch.Tensor) -> dict:
        """Return routing decisions for input batch [B, d]."""
        logits = self.router(h)                            # [B, N]
        topk_vals, topk_idx = logits.topk(self.cfg.top_k, dim=-1)
        # Normalized gating over top-K (standard MoE convention)
        topk_probs = F.softmax(topk_vals, dim=-1)
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
                out[b] += g * self.experts[i](h[b : b + 1]).squeeze(0)
        return out, info


def make_calibration_data(
    moe: ToyMoE, n_samples: int = 512, seed: int = 0
) -> torch.Tensor:
    """Hidden states with mild structure to mimic real activations.

    Mix of N Gaussians (one per expert "topic") to encourage some
    expert specialization in routing.
    """
    g = torch.Generator().manual_seed(seed)
    d = moe.cfg.d_model
    N = moe.cfg.n_experts
    # Pick N centers in d-dim space
    centers = torch.randn(N, d, generator=g) * 0.5
    # Sample assignments
    idx = torch.randint(0, N, (n_samples,), generator=g)
    H = centers[idx] + 0.3 * torch.randn(n_samples, d, generator=g)
    return H
