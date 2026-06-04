"""Experiment 16b — Stage 3 + Stage 4 on cached OLMoE layer 0.

Same as exp 16 but loads from olmoe_layer0_cache.pt (1.5GB) instead
of the full 14GB OLMoE model. Avoids OOM during cert-aware merging.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import torch
import torch.nn as nn

from cert_moe.toy_swiglu_moe import (
    ToySwiGLUMoE, ToySwiGLUMoEConfig, SwiGLUExpert, apply_merge_swiglu,
)
from cert_moe.olmoe_adapter import OLMoEMoEConfig
from cert_moe.merge import (
    complete_linkage_clusters, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions


def build_moe_from_cache(cache):
    """Reconstruct an OLMoE-shaped wrapper using ToySwiGLUMoE classes.

    We reuse ToySwiGLUMoE because it already has SwiGLU forward, plus
    apply_merge_swiglu is compatible. We just initialize with the
    cached weights and disable random init.
    """
    cfg = ToySwiGLUMoEConfig(
        d_model=cache["d_model"],
        d_ff=cache["d_ff"],
        n_experts=cache["n_experts"],
        top_k=cache["top_k"],
        n_clone_pairs=0,
    )
    moe = ToySwiGLUMoE.__new__(ToySwiGLUMoE)
    nn.Module.__init__(moe)
    moe.cfg = cfg
    moe.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
    with torch.no_grad():
        moe.router.weight.copy_(cache["router_weight"])
    moe.experts = nn.ModuleList()
    for i in range(cfg.n_experts):
        e = SwiGLUExpert(cfg.d_model, cfg.d_ff)
        with torch.no_grad():
            e.W1.copy_(cache["experts_W1"][i])
            e.W2.copy_(cache["experts_W2"][i])
            e.W3.copy_(cache["experts_W3"][i])
        moe.experts.append(e)
    moe.clone_pairs = []
    return moe


def cosine_swiglu(experts):
    import torch.nn.functional as F
    vecs = torch.stack([
        torch.cat([e.W1.detach().flatten(),
                    e.W2.detach().flatten(),
                    e.W3.detach().flatten()])
        for e in experts
    ])
    vecs = F.normalize(vecs, dim=-1)
    sim = vecs @ vecs.T              # [N, N] — avoids [N, N, P] broadcast OOM
    dist = 1.0 - sim
    dist.fill_diagonal_(0.0)
    return dist


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist()
                       if math.isfinite(d)))
    for eps in flat:
        plan = complete_linkage_clusters(dist, epsilon=eps)
        if len(plan.clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def free_pgd(moe, merged, h_init, n_steps, step_size, lo, hi):
    h = h_init.clone().detach().requires_grad_(True)
    best = -1.0
    for _ in range(n_steps):
        y_o, _ = moe(h)
        y_m, _ = merged(h)
        loss = -(y_o - y_m).abs().max(-1).values.sum()
        loss.backward()
        with torch.no_grad():
            h += step_size * h.grad.sign()
            h.clamp_(lo, hi)
            h.grad = None
            y_o, _ = moe(h)
            y_m, _ = merged(h)
            best = max(best, (y_o - y_m).abs().max(-1).values.max().item())
        h.requires_grad_(True)
    return best


def cert_pgd(moe, merged, plan, h_init, K, n_steps, step_size, lo, hi):
    W_g = moe.router.weight.detach()
    if not check_conditions(h_init, W_g, plan, K=K,
                             delta_threshold=0.1).all_certified.all():
        return -1.0
    h = h_init.clone().detach()
    best = -1.0
    for _ in range(n_steps):
        h_g = h.clone().requires_grad_(True)
        y_o, _ = moe(h_g)
        y_m, _ = merged(h_g)
        (-(y_o - y_m).abs().max(-1).values.sum()).backward()
        gsign = h_g.grad.sign().detach()
        accepted = False
        for s in (step_size, step_size / 2, step_size / 4):
            with torch.no_grad():
                h_t = (h + s * gsign).clamp(lo, hi)
                if check_conditions(h_t, W_g, plan, K=K,
                                     delta_threshold=0.1).all_certified.all():
                    h = h_t
                    accepted = True
                    break
        if not accepted:
            break
        with torch.no_grad():
            y_o, _ = moe(h)
            y_m, _ = merged(h)
            best = max(best, (y_o - y_m).abs().max(-1).values.max().item())
    return best


def main():
    cache_path = str(
        pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"
    )
    print(f"Loading cache from {cache_path}...")
    cache = torch.load(cache_path, weights_only=True)
    moe = build_moe_from_cache(cache)
    H = cache["H"].clone()
    # Free the cache dict — moe has its own copies of the weights now
    del cache
    import gc; gc.collect()
    print(f"MoE: d={moe.cfg.d_model}, d_ff={moe.cfg.d_ff}, "
          f"N={moe.cfg.n_experts}, K={moe.cfg.top_k}")
    print(f"Calibration H: {tuple(H.shape)}")

    # Routing freq
    with torch.no_grad():
        logits = H @ moe.router.weight.T
    _, topk_idx = logits.topk(moe.cfg.top_k, dim=-1)
    freq = torch.zeros(moe.cfg.n_experts)
    for k in range(moe.cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    print(f"Expert usage range: [{freq.min().item():.0f}, "
          f"{freq.max().item():.0f}], mean {freq.mean().item():.1f}")

    print("\nBuilding cosine distance matrix...")
    d_cos = cosine_swiglu(moe.experts)
    print(f"  d_cos range: [{d_cos[d_cos > 0].min():.4f}, {d_cos.max():.4f}]")

    # We'll do 64→48 (mild) and 64→32 (50%) to see the effect
    for target_n in (48, 32):
        print(f"\n{'='*70}")
        print(f"Compression 64 → {target_n}")
        print(f"{'='*70}")

        # Cosine plan
        eps_c = find_cutoff(d_cos, target_n)
        plan_cos = complete_linkage_clusters(d_cos, epsilon=eps_c)
        merged_cos = apply_merge_swiglu(moe, plan_cos, freq)

        # Cert-aware plan
        print("  building cert-aware plan...", flush=True)
        max_b = d_cos[d_cos.isfinite()].max().item() * 1.001
        plan_ca = cert_aware_greedy_merge(
            moe, H, d_cos,
            target_n_clusters=target_n,
            max_bound=max_b, delta_em=0.1, bound_weight=0.0,
        )
        merged_ca = apply_merge_swiglu(moe, plan_ca, freq)

        # Compute outputs once
        with torch.no_grad():
            y_o, _ = moe(H)
            y_cos, _ = merged_cos(H)
            y_ca, _ = merged_ca(H)
        ref_norm = y_o.abs().mean().item()

        rel_cos = (y_o - y_cos).abs().mean().item() / max(ref_norm, 1e-8)
        rel_ca = (y_o - y_ca).abs().mean().item() / max(ref_norm, 1e-8)

        m_cos = check_conditions(H, moe.router.weight.detach(),
                                  plan_cos, K=moe.cfg.top_k,
                                  delta_threshold=0.1)
        m_ca = check_conditions(H, moe.router.weight.detach(),
                                 plan_ca, K=moe.cfg.top_k,
                                 delta_threshold=0.1)
        print(f"\n  cosine    : rel_div {rel_cos*100:.1f}%, "
              f"MC {m_cos.mc.float().mean()*100:.1f}%, "
              f"TS {m_cos.ts.float().mean()*100:.1f}%, "
              f"EM {m_cos.em.float().mean()*100:.1f}%, "
              f"cert {m_cos.all_certified.float().mean()*100:.1f}%")
        print(f"  cert-aware: rel_div {rel_ca*100:.1f}%, "
              f"MC {m_ca.mc.float().mean()*100:.1f}%, "
              f"TS {m_ca.ts.float().mean()*100:.1f}%, "
              f"EM {m_ca.em.float().mean()*100:.1f}%, "
              f"cert {m_ca.all_certified.float().mean()*100:.1f}%")

        # Adversarial PGD (small budget)
        lo = H.min(0).values - 0.5
        hi = H.max(0).values + 0.5
        step_size = (hi - lo).mean().item() * 0.02
        n_starts = 4
        n_steps = 20
        torch.manual_seed(0)
        starts = (lo + (hi - lo)
                  * torch.rand(n_starts, moe.cfg.d_model)).unsqueeze(1)

        worst_cos = max(
            free_pgd(moe, merged_cos, starts[s], n_steps, step_size, lo, hi)
            for s in range(n_starts)
        )
        worst_ca = max(
            free_pgd(moe, merged_ca, starts[s], n_steps, step_size, lo, hi)
            for s in range(n_starts)
        )
        print(f"  free PGD cosine    = {worst_cos:.3f}")
        print(f"  free PGD cert-aware = {worst_ca:.3f}")

        # Cert-PGD on cert-aware
        cert_pool = H[m_ca.all_certified]
        if cert_pool.shape[0] >= 1:
            n_c = min(n_starts, cert_pool.shape[0])
            worst_cp = max(
                cert_pgd(moe, merged_ca, plan_ca, cert_pool[s:s+1],
                          moe.cfg.top_k, n_steps, step_size, lo, hi)
                for s in range(n_c)
            )
            print(f"  cert-PGD cert-aware = {worst_cp:.3f}")
            if worst_cos > 0:
                red = (worst_cos - worst_cp) / worst_cos * 100
                print(f"  reduction vs cosine free PGD: {red:.1f}%")
        else:
            print(f"  cert pool empty — cert-PGD skipped")


if __name__ == "__main__":
    main()
