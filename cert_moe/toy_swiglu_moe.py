"""Synthetic SwiGLU MoE for Stage 1-4 pipeline validation.

Mirror of toy_moe.py but with SwiGLU experts. Used to verify the
SwiGLU bound (cert_moe.swiglu_bounds) integrates correctly with the
rest of the framework (Stage 1 router stability, cert-aware merging,
Theorem 1 conditions, adversarial PGD).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cert_moe.swiglu_bounds import silu


@dataclass
class ToySwiGLUMoEConfig:
    d_model: int = 64
    d_ff: int = 128
    n_experts: int = 16
    top_k: int = 2
    n_clone_pairs: int = 2
    clone_noise_std: float = 0.005


class SwiGLUExpert(nn.Module):
    """y = W_3 · (SiLU(W_1 h) ⊙ (W_2 h))."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        s = d_model ** -0.5
        self.W1 = nn.Parameter(torch.randn(d_ff, d_model) * s)
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model) * s)
        self.W3 = nn.Parameter(torch.randn(d_model, d_ff) * s)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (silu(h @ self.W1.T) * (h @ self.W2.T)) @ self.W3.T


class ToySwiGLUMoE(nn.Module):
    def __init__(self, cfg: ToySwiGLUMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(cfg.d_model, cfg.d_ff)
             for _ in range(cfg.n_experts)]
        )
        self.clone_pairs = self._plant_clones()

    def _plant_clones(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        n = self.cfg.n_experts
        k = self.cfg.n_clone_pairs
        with torch.no_grad():
            for p in range(k):
                i = n - 2 * (k - p)
                j = i + 1
                pairs.append((i, j))
                noise = self.cfg.clone_noise_std
                for name in ("W1", "W2", "W3"):
                    src = getattr(self.experts[i], name)
                    dst = getattr(self.experts[j], name)
                    dst.copy_(src + noise * torch.randn_like(src))
        return pairs

    def route(self, h: torch.Tensor) -> dict:
        logits = self.router(h)
        topk_vals, topk_idx = logits.topk(self.cfg.top_k, dim=-1)
        topk_probs = F.softmax(topk_vals, dim=-1)
        return {"logits": logits, "topk_idx": topk_idx,
                "topk_probs": topk_probs}

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


def make_calibration_data(
    moe: ToySwiGLUMoE, n_samples: int = 512, seed: int = 0
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    d = moe.cfg.d_model
    N = moe.cfg.n_experts
    centers = torch.randn(N, d, generator=g) * 0.5
    idx = torch.randint(0, N, (n_samples,), generator=g)
    return centers[idx] + 0.3 * torch.randn(n_samples, d, generator=g)


def apply_merge_swiglu(
    moe: ToySwiGLUMoE, plan, routing_freq: torch.Tensor,
) -> "ToySwiGLUMoE":
    """SwiGLU-aware variant of merge.apply_merge (handles W1/W2/W3)."""
    n_new = len(plan.clusters)
    new_cfg = ToySwiGLUMoEConfig(
        d_model=moe.cfg.d_model,
        d_ff=moe.cfg.d_ff,
        n_experts=n_new,
        top_k=min(moe.cfg.top_k, n_new),
        n_clone_pairs=0,
    )
    new_moe = ToySwiGLUMoE.__new__(ToySwiGLUMoE)
    nn.Module.__init__(new_moe)
    new_moe.cfg = new_cfg
    new_moe.router = nn.Linear(new_cfg.d_model, n_new, bias=False)
    new_moe.experts = nn.ModuleList(
        [SwiGLUExpert(new_cfg.d_model, new_cfg.d_ff) for _ in range(n_new)]
    )
    new_moe.clone_pairs = []

    with torch.no_grad():
        for c_idx, members in enumerate(plan.clusters):
            freqs = routing_freq[members]
            if freqs.sum() == 0:
                w = torch.ones_like(freqs) / len(freqs)
            else:
                w = freqs / freqs.sum()
            for name in ("W1", "W2", "W3"):
                avg = sum(w[i] * getattr(moe.experts[m], name)
                          for i, m in enumerate(members))
                getattr(new_moe.experts[c_idx], name).copy_(avg)
            cluster_row = torch.logsumexp(
                moe.router.weight[members], dim=0,
            )
            new_moe.router.weight[c_idx].copy_(cluster_row)

    return new_moe
