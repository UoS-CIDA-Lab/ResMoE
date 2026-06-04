"""Experiment 43 — Per-input certified robustness radius for aliasing.

Exp 42 tightened the alias-error bound 7.6x via the routed zonotope but it was
still ~71x loose over the whole routed CLUSTER (which spans the data spread).
For a per-input ROBUSTNESS guarantee the right region is the small L_inf
eps-BALL around each input, not the cluster. Two facts make the per-input
certificate clean:
  (1) the aliased model and the original SHARE the router, so they differ ONLY
      in the expert FUNCTION of aliased experts; the output deviation at any x is
      exactly the alias error ||E_{top1(x)} - E_{rep(top1(x))}|| (0 if top1 not
      aliased). Routing flips don't add unbounded error (both flip together).
  (2) within an input's routing-stable radius r_route, top1 is fixed = i, so the
      deviation over the eps-ball (eps <= r_route) is just ||E_i - E_rep(i)|| over
      the tiny box [h-eps, h+eps] — TIGHT.

So the per-input certified equivalence-robustness radius is gated by r_route
(MoE-intrinsic routing fragility), and within it the alias-error bound should be
tight (~2-3x, not 71x). This experiment measures both:
  - r_route distribution (certified routing-stable radius per input)
  - alias-error bound looseness over the eps-ball (CROWN vs PGD)

Run: python3 experiments/43_switch_perinput_radius.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import pair_diff_bound, empirical_pair_diff
from cert_moe.router_stability import pairwise_router_l1

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


def expert_out(e, h):
    return torch.relu(h @ e.W1.T) @ e.W2.T


def pgd_pair_diff(ei, ej, h0, eps, n_steps=100):
    """max ||E_i(x)-E_j(x)||_inf over ||x-h0||_inf <= eps."""
    step = eps / 10
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps):
        hg = h.clone().requires_grad_(True)
        d = (expert_out(ei, hg) - expert_out(ej, hg)).abs().max(-1).values.sum()
        d.backward()
        with torch.no_grad():
            h = torch.max(torch.min(h + step * hg.grad.sign(), h0 + eps), h0 - eps)
            cur = (expert_out(ei, h) - expert_out(ej, h)).abs().max(-1).values.max().item()
            best = max(best, cur)
    return best


def routing_stable_radius(h, W_g, l1):
    """Largest r (L_inf) keeping top-1 fixed: min_k (l_top1 - l_k)/||W_top1-W_k||_1."""
    logits = (h @ W_g.T).squeeze(0)             # [N]
    top1 = int(logits.argmax())
    r = float("inf")
    for k in range(W_g.shape[0]):
        if k == top1:
            continue
        denom = l1[top1, k].item()
        if denom > 0:
            r = min(r, (logits[top1] - logits[k]).item() / denom)
    return r, top1


def main():
    print("Loading Switch-base-8 (top-1)...")
    moe, model, tok = load_switch_block()
    N = moe.cfg.n_experts
    W_g = moe.router.weight.detach()
    l1 = pairwise_router_l1(W_g)
    H = collect_hidden_states(model, tok, PATH, SAMPLES, batch_size=8).to(W_g.dtype)
    if H.shape[0] > 512:
        H = H[torch.randperm(H.shape[0])[:512]]
    top1 = (H @ W_g.T).argmax(-1)
    freq = torch.zeros(N).scatter_add_(0, top1, torch.ones(H.shape[0]))
    print(f"  N={N}, calib={tuple(H.shape)}, top-1 freq {freq.int().tolist()}")

    # alias the 2 lowest-freq experts to their empirical-nearest twin (over
    # that expert's routed region). (Selection method is not the focus here.)
    aliased = freq.argsort()[:2].tolist()
    rep_of = list(range(N))
    for i in aliased:
        routed_i = H[top1 == i]
        if routed_i.shape[0] < 4:
            routed_i = H
        cand = [(empirical_pair_diff(moe.experts[i], moe.experts[j], routed_i), j)
                for j in range(N) if j != i]
        rep_of[i] = min(cand)[1]
    print(f"  aliased {aliased} -> {[rep_of[i] for i in aliased]}")

    # routing-stable radius distribution (all inputs)
    rr = torch.tensor([routing_stable_radius(H[b:b+1], W_g, l1)[0]
                       for b in range(H.shape[0])])
    rr = rr[torch.isfinite(rr)]
    qs = torch.quantile(rr, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    print(f"\n  routing-stable radius r_route (L_inf): "
          f"median {rr.median():.3f}, "
          f"q[10,25,50,75,90]={[round(x,3) for x in qs.tolist()]}")

    print(f"\n{'='*68}")
    print("  per-input alias-error bound over eps-ball: CROWN vs PGD (looseness)")
    print(f"{'='*68}")
    print(f"  {'eps':>6s} {'stable%':>8s} {'med cert':>10s} {'med emp':>10s} "
          f"{'looseness':>10s}")
    print("  " + "-" * 56)
    # sample inputs routed to aliased experts
    pool = torch.cat([torch.nonzero(top1 == i).flatten() for i in aliased])
    torch.manual_seed(0)
    pool = pool[torch.randperm(len(pool))[:30]]
    for eps in (0.1, 0.25, 0.5, 1.0):
        certs, emps, n_stable = [], [], 0
        for b in pool.tolist():
            h = H[b:b+1]
            r_route, i = routing_stable_radius(h, W_g, l1)
            j = rep_of[i]
            if j == i:
                continue
            stable = eps <= r_route
            n_stable += int(stable)
            if not stable:
                continue   # within-ball routing may flip; simple cert applies only when stable
            c = pair_diff_bound(moe.experts[i], moe.experts[j],
                                (h - eps).squeeze(0), (h + eps).squeeze(0),
                                method="alpha_crown", alpha_iters=20)
            e = pgd_pair_diff(moe.experts[i], moe.experts[j], h, eps)
            certs.append(float(c)); emps.append(e)
        if certs:
            mc = sorted(certs)[len(certs)//2]
            me = sorted(emps)[len(emps)//2]
            print(f"  {eps:>6.2f} {n_stable/len(pool)*100:>7.1f}% {mc:>10.3f} "
                  f"{me:>10.3f} {mc/max(me,1e-6):>9.1f}x")
        else:
            print(f"  {eps:>6.2f} {n_stable/len(pool)*100:>7.1f}% "
                  f"{'(no routing-stable inputs at this eps)':>32s}")

    print(f"\n{'='*68}")
    print("  READ: looseness over the small eps-ball should be MUCH < 71x")
    print("  (exp42 cluster) -> tight per-input alias guarantee. stable% shows")
    print("  the radius is gated by routing stability (MoE-intrinsic), not by")
    print("  the alias bound. Within r_route, output dev <= cert (provable).")


if __name__ == "__main__":
    main()
