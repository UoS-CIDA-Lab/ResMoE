"""Experiment 41 — Certified equivalence-based expert ALIASING (router preserved).

The corrected form of the project's original thesis. Prune expert i by ALIASING
it to a kept expert j whose output is CERTIFIED equivalent over the input region:
the router is left UNTOUCHED (it still selects i, with i's gate), but i's weights
are dropped and E_j is computed instead. Routing is preserved exactly (no TS
fragility); the only error is ||E_i(x) - E_j(x)||, which is EXACTLY what the
cert_moe expert-difference bound (CROWN) certifies. So verification targets the
right quantity and the bound IS the guarantee:
    alias i->j allowed iff  max_{x in R} ||E_i(x)-E_j(x)|| <= delta   (certified)
=> over R (an eps-inflated data box, i.e. an adversarial eps-ball region), the
   aliased model deviates from the original by <= delta, PROVABLY — even under
   adversarial perturbation. Frequency pruning reroutes a dropped expert's tokens
   to an arbitrary (possibly far) runner-up: no bound, silent failures.

Resolves exp 24 (output-equivalence was wrong for weight-AVERAGING; it is RIGHT
for aliasing) and exp 25 (aliasing failed only because the router was still
merged; here it is preserved).

Switch-base-8, top-1. Reports: (1) pairwise certified delta matrix + nearest
twin per expert, (2) aliasing budget curve (#prunable vs delta_tol), (3) at a
matched prune count, certified-aliasing vs frequency-pruning under eps-PGD.

Run: python3 experiments/41_switch_certified_aliasing.py
"""
from __future__ import annotations

import sys
import pathlib
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import pair_diff_bound

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

EPS = 0.5            # region inflation = adversarial L_inf budget


def switch_forward(moe, h, rep_of=None, drop_mask=None):
    """Top-1 Switch output. rep_of[i] = expert function to compute when the
    router selects i (aliasing). drop_mask[i]=True removes i from the router."""
    logits = h @ moe.router.weight.T
    if drop_mask is not None:
        logits = logits.masked_fill(drop_mask.unsqueeze(0), float("-inf"))
    probs = F.softmax(logits, dim=-1)
    top1 = logits.argmax(-1)
    gate = probs.gather(1, top1.unsqueeze(1)).squeeze(1)
    y = torch.zeros_like(h)
    for i in torch.unique(top1).tolist():
        m = top1 == i
        used = rep_of[i] if rep_of is not None else i
        e = moe.experts[used]
        y[m] = (torch.relu(h[m] @ e.W1.T) @ e.W2.T) * gate[m].unsqueeze(1)
    return y


def certified_delta_matrix(moe, lo, hi):
    N = moe.cfg.n_experts
    D = torch.zeros(N, N)
    for i, j in combinations(range(N), 2):
        b = pair_diff_bound(moe.experts[i], moe.experts[j], lo, hi, method="crown")
        D[i, j] = D[j, i] = float(b)
    return D


def aliasing_plan(D, freq, delta_tol):
    """Greedy: alias low-freq experts to a kept rep within certified delta_tol.
    Returns rep_of (list), aliased (set of pruned experts)."""
    N = D.shape[0]
    rep_of = list(range(N))
    is_rep = [True] * N
    aliased = []
    for i in freq.argsort().tolist():          # low-freq first
        cands = [(D[i, j].item(), j) for j in range(N)
                 if j != i and is_rep[j] and not (j in aliased)]
        cands = [(d, j) for d, j in cands if d <= delta_tol]
        if cands:
            d, j = min(cands)
            rep_of[i] = j
            is_rep[i] = False
            aliased.append(i)
    return rep_of, aliased


def pgd_dev(moe, h0, eps, rep_of=None, drop_mask=None, n_steps=120):
    step = eps / 12
    h = h0.clone().detach()
    best = 0.0
    for _ in range(n_steps):
        hg = h.clone().requires_grad_(True)
        d = (switch_forward(moe, hg) - switch_forward(moe, hg, rep_of, drop_mask)
             ).abs().max(-1).values.sum()
        d.backward()
        with torch.no_grad():
            h = torch.max(torch.min(h + step * hg.grad.sign(), h0 + eps), h0 - eps)
            cur = (switch_forward(moe, h)
                   - switch_forward(moe, h, rep_of, drop_mask)
                   ).abs().max(-1).values.max().item()
            best = max(best, cur)
    return best


def main():
    print("Loading Switch-base-8 (top-1)...")
    moe, model, tok = load_switch_block()
    N = moe.cfg.n_experts
    H = collect_hidden_states(model, tok, PATH, SAMPLES, batch_size=8
                              ).to(moe.router.weight.dtype)
    if H.shape[0] > 512:
        H = H[torch.randperm(H.shape[0])[:512]]
    logits = H @ moe.router.weight.T
    top1 = logits.argmax(-1)
    freq = torch.zeros(N).scatter_add_(0, top1, torch.ones(H.shape[0]))
    print(f"  N={N}, calib={tuple(H.shape)}, ||h|| mean {H.norm(dim=-1).mean():.1f}")
    print(f"  top-1 freq: {freq.int().tolist()}")

    lo, hi = H.min(0).values - EPS, H.max(0).values + EPS
    print(f"\nCertified pairwise ||E_i - E_j|| (CROWN, over data box +-{EPS})...")
    D = certified_delta_matrix(moe, lo, hi)
    # empirical output scale for context
    with torch.no_grad():
        ey = torch.stack([torch.relu(H @ moe.experts[i].W1.T) @ moe.experts[i].W2.T
                          for i in range(N)])           # [N,B,d]
    out_scale = ey.abs().max(-1).values.mean().item()
    print(f"  (mean per-expert output |y|_inf ~ {out_scale:.2f})")
    print("  nearest certified twin per expert (min_j delta_ij):")
    for i in range(N):
        offdiag = [(D[i, j].item(), j) for j in range(N) if j != i]
        d, j = min(offdiag)
        print(f"    E{i} -> E{j}: delta={d:.3f}")

    print(f"\n{'='*64}\n  Aliasing budget curve (router preserved)\n{'='*64}")
    print(f"  {'delta_tol':>10s} {'#aliasable':>11s} {'aliased experts'}")
    finite = sorted(set(D[i, j].item() for i, j in combinations(range(N), 2)))
    for dt in finite[:6] + [finite[len(finite)//2], finite[-1]]:
        rep_of, aliased = aliasing_plan(D, freq, dt)
        print(f"  {dt:>10.3f} {len(aliased):>11d} {aliased}")

    # operating point: smallest delta_tol that aliases >=2 experts
    chosen = None
    for dt in finite:
        rep_of, aliased = aliasing_plan(D, freq, dt)
        if len(aliased) >= 2:
            chosen = (dt, rep_of, aliased)
            break
    if chosen is None:
        dt = finite[-1]
        rep_of, aliased = aliasing_plan(D, freq, dt)
        chosen = (dt, rep_of, aliased)
    dt, rep_of, aliased = chosen
    C = len(aliased)
    print(f"\n{'='*64}")
    print(f"  Operating point: certified-alias {C} experts at delta_tol={dt:.3f}")
    print(f"  aliased {aliased}, rep_of={[rep_of[i] for i in aliased]}")
    cert_bound = max(D[i, rep_of[i]].item() for i in aliased)
    print(f"  CERTIFIED worst-case deviation (aliasing) <= {cert_bound:.3f}")

    # frequency pruning at matched count
    drop = torch.zeros(N, dtype=torch.bool)
    drop[freq.argsort()[:C]] = True
    print(f"  frequency-pruned experts: {drop.nonzero().flatten().tolist()}")

    # clean deviation
    with torch.no_grad():
        y0 = switch_forward(moe, H)
        dev_alias = (y0 - switch_forward(moe, H, rep_of=rep_of)
                     ).abs().max(-1).values.max().item()
        dev_freq = (y0 - switch_forward(moe, H, drop_mask=drop)
                    ).abs().max(-1).values.max().item()
    print(f"\n  clean worst dev (calib):  aliasing {dev_alias:.3f} | "
          f"frequency {dev_freq:.3f}")

    # eps-PGD worst-case deviation
    torch.manual_seed(0)
    starts = torch.randperm(H.shape[0])[:12]
    pgd_alias = max(pgd_dev(moe, H[i:i+1], EPS, rep_of=rep_of) for i in starts)
    pgd_freq = max(pgd_dev(moe, H[i:i+1], EPS, drop_mask=drop) for i in starts)
    print(f"  eps-PGD worst dev (eps={EPS}): aliasing {pgd_alias:.3f} "
          f"(cert bound {cert_bound:.3f}) | frequency {pgd_freq:.3f} (no bound)")

    print(f"\n{'='*64}")
    print("  KEY: aliasing eps-PGD dev should stay <= certified bound (provably")
    print("  robust); frequency-pruning dev unbounded = silent adversarial")
    print("  failure. Budget curve shows how many experts have a certified twin.")


if __name__ == "__main__":
    main()
