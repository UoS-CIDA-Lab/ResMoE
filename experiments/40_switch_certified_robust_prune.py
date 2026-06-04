"""Experiment 40 (POC) — Certified-robust expert pruning vs frequency pruning
under adversarial attack (Switch-base-8, top-1).

Positioning: routing-orthogonal compression (prune experts, KEEP the router)
yields a model whose discrete top-K routing — the real adversarial surface of
an MoE — is CERTIFIABLY stable. We prune only experts that are PROVABLY never
the argmax within an L_inf ball of radius eps around the data (a "certified-
dead" set), so within that ball the pruned model is provably IDENTICAL to the
original. Frequency pruning has no such guarantee: it drops low-usage experts
that an adversary can still route into, causing silent output jumps.

Mechanism (stated to avoid fooling ourselves): the asset is CERTIFIED ROUTING
STABILITY (router_stability.stable_topk_mask), not gradient masking. We report
the FORMAL certificate (router margins) as primary evidence and PGD as
corroboration.

For radius eps, expert i is "eps-dead" if for EVERY calibration input h the
current argmax m satisfies  l_m(h) - l_i(h) >= eps * ||W_m - W_i||_1  (m provably
stays above i under any ||delta||_inf <= eps -> i never becomes top-1). Dropping
eps-dead experts is lossless over every eps-ball around the data.

Compares, at matched prune counts: certified-pruned vs frequency-pruned.
Metrics: (1) #eps-dead experts (certified budget), (2) certified routing-stable
fraction, (3) worst-case output deviation from the original under eps-PGD.

Run: python3 experiments/40_switch_certified_robust_prune.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.router_stability import stable_topk_mask, pairwise_router_l1

PATH = "encoder.block.1.layer.1.mlp"
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
] * 6


def switch_forward(moe, h, drop_mask=None):
    """Top-1 Switch output: gate(softmax prob of argmax) * expert(h).
    drop_mask: [N] bool, True = pruned (router logit set to -inf)."""
    logits = h @ moe.router.weight.T                       # [B, N]
    if drop_mask is not None:
        logits = logits.masked_fill(drop_mask.unsqueeze(0), float("-inf"))
    probs = F.softmax(logits, dim=-1)
    top1 = logits.argmax(-1)                               # [B]
    gate = probs.gather(1, top1.unsqueeze(1)).squeeze(1)   # [B]
    y = torch.zeros_like(h)
    for i in torch.unique(top1).tolist():
        m = top1 == i
        e = moe.experts[i]
        y[m] = (torch.relu(h[m] @ e.W1.T) @ e.W2.T) * gate[m].unsqueeze(1)
    return y


def eps_dead_experts(H, W_g, eps):
    """Experts provably never argmax within ||delta||_inf <= eps of any h in H.
    i is eps-dead if for all b: l_{argmax}(h_b) - l_i(h_b) >= eps*||W_argmax-W_i||_1."""
    logits = H @ W_g.T                                     # [B, N]
    N = W_g.shape[0]
    l1 = pairwise_router_l1(W_g)                           # [N, N]
    top1 = logits.argmax(-1)                               # [B]
    margin = logits.gather(1, top1.unsqueeze(1)) - logits  # [B, N] l_m - l_i
    required = eps * l1[top1]                               # [B, N] ||W_m - W_i||_1*eps
    # i is certified-out at h_b if margin >= required (m stays above i)
    cert_out = margin >= required                          # [B, N]
    # but the current argmax itself has margin 0; exclude experts that ARE argmax somewhere
    is_argmax_somewhere = torch.zeros(N, dtype=torch.bool)
    is_argmax_somewhere[top1.unique()] = True
    dead = cert_out.all(dim=0) & (~is_argmax_somewhere)
    return dead                                            # [N] bool


def pgd_deviation(moe, drop_mask, h0, eps, n_steps=100, step=None):
    """Max_inf ||orig(h)-pruned(h)|| over ||h-h0||_inf <= eps (PGD)."""
    if step is None:
        step = eps / 10
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps):
        hg = h.clone().requires_grad_(True)
        y_o = switch_forward(moe, hg, None)
        y_p = switch_forward(moe, hg, drop_mask)
        loss = (y_o - y_p).abs().max(-1).values.sum()
        loss.backward()
        with torch.no_grad():
            h = h + step * hg.grad.sign()
            h = torch.max(torch.min(h, h0 + eps), h0 - eps)  # L_inf ball
            y_o = switch_forward(moe, h, None)
            y_p = switch_forward(moe, h, drop_mask)
            best = max(best, (y_o - y_p).abs().max(-1).values.max().item())
    return best


def main():
    print("Loading Switch-base-8 (top-1)...")
    moe, model, tok = load_switch_block()
    W_g = moe.router.weight.detach()
    N, K = moe.cfg.n_experts, moe.cfg.top_k
    H = collect_hidden_states(model, tok, PATH, SAMPLES, batch_size=8).to(W_g.dtype)
    if H.shape[0] > 512:
        H = H[torch.randperm(H.shape[0])[:512]]
    print(f"  N={N}, top_k={K}, calib H={tuple(H.shape)}, "
          f"||h|| mean {H.norm(dim=-1).mean():.1f}")

    logits = H @ W_g.T
    top1 = logits.argmax(-1)
    freq = torch.zeros(N).scatter_add_(0, top1, torch.ones(H.shape[0]))
    print(f"  expert top-1 freq: {freq.int().tolist()}")

    # eps sweep relative to hidden-state scale
    eps_list = [0.1, 0.25, 0.5, 1.0]
    print(f"\n{'='*76}")
    print(f"  {'eps':>6s} {'eps-dead':>9s} {'cert-stable%':>13s} "
          f"{'certPGD dev':>12s} {'freqPGD dev':>12s}")
    print(f"  {'':>6s} {'experts':>9s} {'(top-1 @eps)':>13s} "
          f"{'(matched)':>12s} {'(matched)':>12s}")
    print("  " + "-" * 70)

    n_starts = 12
    torch.manual_seed(0)
    start_idx = torch.randperm(H.shape[0])[:n_starts]

    for eps in eps_list:
        dead = eps_dead_experts(H, W_g, eps)
        n_dead = int(dead.sum())

        # certified routing-stable fraction (top-1 stays top-1 under eps)
        _, stable = stable_topk_mask(H, W_g, eps, K)
        cert_stable_frac = stable.gather(1, top1.unsqueeze(1)).float().mean().item()

        if n_dead == 0:
            print(f"  {eps:>6.2f} {n_dead:>9d} {cert_stable_frac*100:>12.1f}% "
                  f"{'--':>12s} {'--':>12s}  (nothing certifiably droppable)")
            continue

        # certified-pruned drop set = eps-dead experts
        cert_drop = dead.clone()
        # frequency-pruned: drop the n_dead LOWEST-frequency experts (matched count)
        freq_drop = torch.zeros(N, dtype=torch.bool)
        freq_drop[freq.argsort()[:n_dead]] = True

        # worst-case eps-PGD deviation from original, over data-centered balls
        cert_dev = max(pgd_deviation(moe, cert_drop, H[i:i+1], eps)
                       for i in start_idx)
        freq_dev = max(pgd_deviation(moe, freq_drop, H[i:i+1], eps)
                       for i in start_idx)

        print(f"  {eps:>6.2f} {n_dead:>9d} {cert_stable_frac*100:>12.1f}% "
              f"{cert_dev:>12.4f} {freq_dev:>12.4f}")

    print(f"\n{'='*76}")
    print("  certPGD dev should be ~0 (dropped experts provably never argmax")
    print("  within eps -> pruned == original); freqPGD dev > 0 = silent")
    print("  failures an adversary triggers in frequency pruning. The formal")
    print("  certificate (eps-dead set) is the primary evidence, PGD corroborates.")


if __name__ == "__main__":
    main()
