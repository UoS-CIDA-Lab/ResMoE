"""Experiment 11 — Multi-layer analysis on Switch-base-8.

The reviewer's first question: "Why only one layer?"
Answer: now we run Stage 1 + Stage 3 + Stage 4 on every encoder MoE
layer and report variance.

Encoder MoE blocks in Switch-base-8: blocks 1, 3, 5, 7, 9, 11 (6 layers).
Decoder blocks need a separate forward driver and are queued for next
session.

For each encoder MoE layer we compute:
  - Stage 1: stable% at r=0.05 (single point — full curve is overkill here)
  - Stage 3: cert-aware vs cosine at 8→4
  - Stage 4: free PGD worst-case + cert-PGD worst-case
  - Cluster composition

Output: per-layer table + aggregate statistics.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
from itertools import combinations

import torch

from cert_moe.switch_adapter import (
    load_switch_block, collect_hidden_states, SwitchMoEWrapper,
    SwitchMoEConfig,
)
from cert_moe.router_stability import stable_topk_mask
from cert_moe.expert_bounds import pair_diff_bound
from cert_moe.merge import (
    complete_linkage_clusters, apply_merge, cert_aware_greedy_merge,
)
from cert_moe.theorem1_conditions import check_conditions
from cert_moe.baselines import cosine_distance_matrix


# Same calibration corpus as experiments 07–10 for consistency
SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "Machine learning models predict outputs from inputs.",
    "Neural networks consist of layers of neurons.",
    "Transformers use attention mechanisms.",
    "Mixture of experts increases model capacity.",
    "Quantum computers may revolutionize computation.",
    "Climate change is a pressing global concern.",
    "Mathematics describes the universe.",
    "Music transcends language barriers.",
    "Education shapes future generations.",
    "Technology drives social change.",
    "Sports build character and camaraderie.",
    "Art expresses what words cannot.",
    "History informs our present and future.",
] * 4   # 64 sentences → ~1024 tokens


def wrap_block_from_model(model, path):
    obj = model
    for piece in path.split("."):
        if piece.isdigit():
            obj = obj[int(piece)]
        else:
            obj = getattr(obj, piece)
    cfg = SwitchMoEConfig()
    return SwitchMoEWrapper(obj, cfg)


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


def analyze_layer(model, tok, path, target_n=4):
    """Run the full pipeline on one MoE layer and return a results dict."""
    moe = wrap_block_from_model(model, path)
    H = collect_hidden_states(model, tok, path, SAMPLES, batch_size=8)
    H = H.to(moe.router.weight.dtype)
    if H.shape[0] > 1024:
        H = H[torch.randperm(H.shape[0])[:1024]]

    # Stage 1
    _, stable = stable_topk_mask(H, moe.router.weight.detach(),
                                  r=0.05, K=moe.cfg.top_k)
    stable_05 = stable.float().sum().item() / max(
        H.shape[0] * moe.cfg.top_k, 1)

    # Usage
    with torch.no_grad():
        logits = H @ moe.router.weight.T
    _, topk_idx = logits.topk(moe.cfg.top_k, dim=-1)
    freq = torch.zeros(moe.cfg.n_experts)
    for k in range(moe.cfg.top_k):
        freq.scatter_add_(0, topk_idx[:, k], torch.ones(H.shape[0]))
    max_usage = freq.max().item()
    min_usage = freq.min().item()
    dump_ratio = max_usage / max(freq.mean().item(), 1.0)

    # Stage 3: cosine vs cert-aware at 8→target_n
    d_cos = cosine_distance_matrix(moe.experts)
    eps_c = find_cutoff(d_cos, target_n)
    plan_cos = complete_linkage_clusters(d_cos, epsilon=eps_c)
    merged_cos = apply_merge(moe, plan_cos, freq)

    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001
    plan_ca = cert_aware_greedy_merge(
        moe, H, d_cos, target_n_clusters=target_n,
        max_bound=max_b, delta_em=0.1, bound_weight=0.0,
    )
    merged_ca = apply_merge(moe, plan_ca, freq)

    # Divergence on calibration
    with torch.no_grad():
        y_o, _ = moe(H)
        y_cos, _ = merged_cos(H)
        y_ca, _ = merged_ca(H)
    div_cos = (y_o - y_cos).abs().mean().item()
    div_ca = (y_o - y_ca).abs().mean().item()
    rel_cos = div_cos / y_o.abs().mean().item()
    rel_ca = div_ca / y_o.abs().mean().item()

    # Cert fraction
    masks_cos = check_conditions(H, moe.router.weight.detach(), plan_cos,
                                  K=moe.cfg.top_k, delta_threshold=0.1)
    masks_ca = check_conditions(H, moe.router.weight.detach(), plan_ca,
                                 K=moe.cfg.top_k, delta_threshold=0.1)
    cert_cos = masks_cos.all_certified.float().mean().item()
    cert_ca = masks_ca.all_certified.float().mean().item()

    # Stage 4: adversarial PGD (smaller budget for speed across 6 layers)
    lo = H.min(0).values - 0.5
    hi = H.max(0).values + 0.5
    step_size = (hi - lo).mean().item() * 0.02

    n_starts = 4
    n_steps_free = 40
    n_steps_cert = 40
    torch.manual_seed(0)
    starts = (lo + (hi - lo)
              * torch.rand(n_starts, moe.cfg.d_model)).unsqueeze(1)

    free_cos = max(free_pgd(moe, merged_cos, starts[s],
                              n_steps_free, step_size, lo, hi)
                    for s in range(n_starts))
    free_ca = max(free_pgd(moe, merged_ca, starts[s],
                             n_steps_free, step_size, lo, hi)
                   for s in range(n_starts))

    # Cert-PGD: only valid for cert-aware (cosine cert pool too small often)
    cert_pool = H[masks_ca.all_certified]
    cert_worst = -1.0
    if cert_pool.shape[0] >= 4:
        for s in range(min(n_starts, cert_pool.shape[0])):
            d = cert_pgd(moe, merged_ca, plan_ca,
                          cert_pool[s:s+1], moe.cfg.top_k,
                          n_steps_cert, step_size, lo, hi)
            cert_worst = max(cert_worst, d)

    return {
        "path": path,
        "stable_05": stable_05,
        "dump_ratio": dump_ratio,
        "max_usage": int(max_usage),
        "rel_cos": rel_cos,
        "rel_ca": rel_ca,
        "cert_cos": cert_cos,
        "cert_ca": cert_ca,
        "free_cos": free_cos,
        "free_ca": free_ca,
        "cert_worst": cert_worst,
        "plan_cos": plan_cos.clusters,
        "plan_ca": plan_ca.clusters,
    }


def main():
    print("Loading Switch-base-8...")
    _, model, tok = load_switch_block()  # only need model + tok

    # Encoder MoE blocks
    layer_paths = [f"encoder.block.{i}.layer.1.mlp"
                   for i in (1, 3, 5, 7, 9, 11)]

    print(f"\nAnalyzing {len(layer_paths)} encoder MoE blocks "
          f"(compression 8→4)...")

    results = []
    for i, path in enumerate(layer_paths):
        print(f"\n  [{i+1}/{len(layer_paths)}] {path}", flush=True)
        try:
            r = analyze_layer(model, tok, path, target_n=4)
            results.append(r)
            print(f"    stable@0.05={r['stable_05']*100:.1f}%, "
                  f"dump_ratio={r['dump_ratio']:.1f}, "
                  f"cert_cos={r['cert_cos']*100:.1f}%, "
                  f"cert_ca={r['cert_ca']*100:.1f}%")
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    # Summary tables
    print(f"\n{'='*98}")
    print("Per-layer summary (8 → 4 compression)")
    print(f"{'='*98}")
    header = (f"{'block':<25s} | {'stab%':>6s} | {'dump':>5s} | "
              f"{'rel cos':>7s} | {'rel ca':>6s} | "
              f"{'cert cos%':>9s} | {'cert ca%':>8s} | "
              f"{'free cos':>8s} | {'free ca':>7s} | {'cert-PGD':>8s}")
    print(header)
    print("-" * len(header))
    for r in results:
        block = r['path'].split('.')[2]
        print(f"  block {block:<19s} | "
              f"{r['stable_05']*100:>5.1f}% | "
              f"{r['dump_ratio']:>5.1f} | "
              f"{r['rel_cos']*100:>6.1f}% | "
              f"{r['rel_ca']*100:>5.1f}% | "
              f"{r['cert_cos']*100:>8.1f}% | "
              f"{r['cert_ca']*100:>7.1f}% | "
              f"{r['free_cos']:>8.1f} | "
              f"{r['free_ca']:>7.1f} | "
              f"{r['cert_worst']:>8.1f}")

    # Aggregate stats
    if results:
        def avg(key):
            vs = [r[key] for r in results if r[key] >= 0]
            return sum(vs) / len(vs) if vs else 0.0

        def std(key):
            vs = [r[key] for r in results if r[key] >= 0]
            m = sum(vs) / len(vs) if vs else 0.0
            return (sum((v - m) ** 2 for v in vs) / max(len(vs)-1, 1))**0.5

        print(f"\n{'='*98}")
        print("Aggregate across encoder MoE blocks")
        print(f"{'='*98}")
        print(f"  stable@0.05    : mean {avg('stable_05')*100:.1f}% ± "
              f"{std('stable_05')*100:.1f}%")
        print(f"  cert cos       : mean {avg('cert_cos')*100:.1f}% ± "
              f"{std('cert_cos')*100:.1f}%")
        print(f"  cert ca        : mean {avg('cert_ca')*100:.1f}% ± "
              f"{std('cert_ca')*100:.1f}%")
        print(f"  free PGD cos   : mean {avg('free_cos'):.2f} ± "
              f"{std('free_cos'):.2f}")
        print(f"  free PGD ca    : mean {avg('free_ca'):.2f} ± "
              f"{std('free_ca'):.2f}")
        print(f"  cert-PGD       : mean {avg('cert_worst'):.2f} ± "
              f"{std('cert_worst'):.2f}")

        # Reductions
        reductions = []
        for r in results:
            if r['free_cos'] > 0 and r['cert_worst'] >= 0:
                red = (r['free_cos'] - r['cert_worst']) / r['free_cos'] * 100
                reductions.append(red)
        if reductions:
            print(f"\n  Adversarial worst-case reduction (cert-PGD vs "
                  f"cosine free-PGD):")
            print(f"    mean {sum(reductions)/len(reductions):.1f}% ± "
                  f"{(sum((r-sum(reductions)/len(reductions))**2 for r in reductions)/max(len(reductions)-1,1))**0.5:.1f}%")
            print(f"    range {min(reductions):.1f}% to "
                  f"{max(reductions):.1f}%")


if __name__ == "__main__":
    main()
