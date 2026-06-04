"""Experiment 16 — Stage 3 + Stage 4 on OLMoE layer 0.

Following the recommendation: skip the expensive C(64,2)=2016 bound
computation, use cosine + cert-aware for Stage 3, then adversarial PGD
for Stage 4. Goal: end-to-end framework validation on a real SwiGLU
MoE, even with limited calibration data (~70 tokens).

Compression: 64 → 32 (50%).
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math

import torch

from cert_moe.olmoe_adapter import (
    load_olmoe_block, collect_hidden_states_olmoe,
)
from cert_moe.toy_swiglu_moe import apply_merge_swiglu
from cert_moe.merge import (
    complete_linkage_clusters, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions
from cert_moe.baselines import cosine_distance_matrix


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


def cosine_swiglu(experts) -> torch.Tensor:
    """Same as cosine_distance_matrix but over W1+W2+W3."""
    import torch.nn.functional as F
    N = len(experts)
    vecs = torch.stack([
        torch.cat([e.W1.detach().flatten(),
                    e.W2.detach().flatten(),
                    e.W3.detach().flatten()])
        for e in experts
    ])
    sim = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
    dist = 1.0 - sim
    dist.fill_diagonal_(0.0)
    return dist


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
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
    print("Loading OLMoE layer 0...")
    moe, model, tok = load_olmoe_block(layer_idx=0)
    print(f"  N={moe.cfg.n_experts}, K={moe.cfg.top_k}, "
          f"d={moe.cfg.d_model}, d_ff={moe.cfg.d_ff}")

    print("\nCollecting calibration (slow patched forward)...")
    H = collect_hidden_states_olmoe(
        model, tok, layer_idx=0, texts=SAMPLES, batch_size=2,
    ).to(moe.router.weight.dtype)
    if H.shape[0] > 256:
        H = H[torch.randperm(H.shape[0])[:256]]
    print(f"  Got {H.shape[0]} tokens, ||h||_2 mean {H.norm(dim=-1).mean().item():.2f}")

    # Free up memory: drop the full model
    del model
    import gc; gc.collect()

    # Routing freq
    with torch.no_grad():
        logits = H @ moe.router.weight.T
    _, topk_idx = logits.topk(moe.cfg.top_k, dim=-1)
    freq = torch.zeros(moe.cfg.n_experts)
    for k in range(moe.cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))

    print("\nBuilding cosine distance matrix...")
    d_cos = cosine_swiglu(moe.experts)

    target_n = 32
    print(f"\nCompression 64 → {target_n}")

    # Cosine plan
    eps_c = find_cutoff(d_cos, target_n)
    plan_cos = complete_linkage_clusters(d_cos, epsilon=eps_c)
    merged_cos = apply_merge_swiglu(moe, plan_cos, freq)
    print(f"  cosine plan: {len(plan_cos.clusters)} clusters")

    # Cert-aware plan
    print("  building cert-aware plan (may take a while)...", flush=True)
    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001
    plan_ca = cert_aware_greedy_merge(
        moe, H, d_cos,
        target_n_clusters=target_n,
        max_bound=max_b, delta_em=0.1, bound_weight=0.0,
    )
    merged_ca = apply_merge_swiglu(moe, plan_ca, freq)
    print(f"  cert-aware plan: {len(plan_ca.clusters)} clusters")

    # ============ Measure cert% and divergence ============
    print("\n--- Measurements at 64→32 ---")
    with torch.no_grad():
        y_o, _ = moe(H)
    print(f"  Forwarded {H.shape[0]} tokens through original. "
          f"Output ||y||_2 mean {y_o.norm(dim=-1).mean().item():.2f}")

    rows = []
    for name, plan, merged in [("cosine", plan_cos, merged_cos),
                                ("cert-aware", plan_ca, merged_ca)]:
        with torch.no_grad():
            y_m, _ = merged(H)
        rel_div = (y_o - y_m).abs().mean().item() / y_o.abs().mean().item()
        masks = check_conditions(H, moe.router.weight.detach(), plan,
                                  K=moe.cfg.top_k, delta_threshold=0.1)
        rows.append({
            "name": name, "rel_div": rel_div,
            "mc": masks.mc.float().mean().item(),
            "ts": masks.ts.float().mean().item(),
            "em": masks.em.float().mean().item(),
            "cert": masks.all_certified.float().mean().item(),
            "plan": plan, "merged": merged,
        })
        print(f"  {name:11s}: rel_div {rel_div*100:.1f}%, "
              f"MC {masks.mc.float().mean()*100:.1f}%, "
              f"TS {masks.ts.float().mean()*100:.1f}%, "
              f"EM {masks.em.float().mean()*100:.1f}%, "
              f"cert {masks.all_certified.float().mean()*100:.1f}%")

    # ============ Adversarial PGD ============
    print(f"\n--- Adversarial PGD (limited budget) ---")
    lo = H.min(0).values - 0.5
    hi = H.max(0).values + 0.5
    step_size = (hi - lo).mean().item() * 0.02
    n_starts = 4
    n_steps = 20
    print(f"  {n_starts} starts × {n_steps} steps, step_size {step_size:.3f}")

    torch.manual_seed(0)
    starts = (lo + (hi - lo)
              * torch.rand(n_starts, moe.cfg.d_model)).unsqueeze(1)

    for r in rows:
        merged = r["merged"]
        worst = -1.0
        for s in range(n_starts):
            d = free_pgd(moe, merged, starts[s], n_steps, step_size, lo, hi)
            worst = max(worst, d)
        r["free_pgd"] = worst
        print(f"  {r['name']:11s} free PGD worst = {worst:.3f}")

    # Cert-PGD on cert-aware (the more promising one)
    plan_ca = rows[1]["plan"]
    merged_ca = rows[1]["merged"]
    m = check_conditions(H, moe.router.weight.detach(),
                          plan_ca, K=moe.cfg.top_k, delta_threshold=0.1)
    cert_pool = H[m.all_certified]
    print(f"\n  cert-aware cert pool: {cert_pool.shape[0]}/{H.shape[0]}")
    if cert_pool.shape[0] >= 2:
        n_c = min(n_starts, cert_pool.shape[0])
        worst_cp = max(
            cert_pgd(moe, merged_ca, plan_ca, cert_pool[s:s+1],
                      moe.cfg.top_k, n_steps, step_size, lo, hi)
            for s in range(n_c)
        )
        print(f"  cert-aware cert-PGD worst = {worst_cp:.3f}")
        if rows[0]["free_pgd"] > 0:
            red = (rows[0]["free_pgd"] - worst_cp) / rows[0]["free_pgd"] * 100
            print(f"  reduction vs cosine free PGD: {red:.1f}%")


if __name__ == "__main__":
    main()
