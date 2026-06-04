"""Experiment 42 — Tighten the certified aliasing bound via the ROUTED zonotope.

Exp 41 showed equivalence-based aliasing >> frequency pruning on clean quality,
BUT the CROWN certificate over the full data box was ~1000x loose (delta ~120000
vs actual ~108) -> useless as a guarantee. Diagnosis: expert i is only selected
in a small sub-region (where it is argmax), not the whole box; bounding over the
huge box is the looseness source.

Fix (exp 23's routed-zonotope, now applied to ALIASING): certify the alias error
||E_i - E_j|| over the ZONOTOPE of inputs ROUTED to i (a PCA hull of H[top1==i],
inflated by margin) — the semantically correct region (i is aliased to j only
where i is selected). Asymmetric: the i->j bound uses i's routed region.

Reports: (1) looseness of full-box-alphaCROWN vs routed-zono bound (does it get
tight enough to be a real guarantee?), (2) aliasing budget curve under the tight
bound, (3) at a matched count, certified bound vs clean vs eps-PGD deviation,
aliasing vs frequency.

Run: python3 experiments/42_switch_routed_zono_aliasing.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cert_moe.switch_adapter import load_switch_block, collect_hidden_states
from cert_moe.expert_bounds import (
    pair_diff_bound, empirical_pair_diff,
    zonotope_from_data, alpha_crown_zonotope_bound,
)

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
EPS = 0.5
MARGIN = 0.5


def switch_forward(moe, h, rep_of=None, drop_mask=None):
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


def aliasing_plan(D, freq, delta_tol):
    """Greedy: alias low-freq i to a kept rep j with certified D[i,j] <= tol."""
    N = D.shape[0]
    rep_of = list(range(N))
    is_rep = [True] * N
    aliased = []
    for i in freq.argsort().tolist():
        cands = [(D[i, j].item(), j) for j in range(N)
                 if j != i and is_rep[j] and j not in aliased and
                 torch.isfinite(D[i, j])]
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
        d = (switch_forward(moe, hg)
             - switch_forward(moe, hg, rep_of, drop_mask)).abs().max(-1).values.sum()
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
    top1 = (H @ moe.router.weight.T).argmax(-1)
    freq = torch.zeros(N).scatter_add_(0, top1, torch.ones(H.shape[0]))
    print(f"  N={N}, calib={tuple(H.shape)}, top-1 freq {freq.int().tolist()}")

    lo_full, hi_full = H.min(0).values - MARGIN, H.max(0).values + MARGIN

    INF = float("inf")
    D_box = torch.full((N, N), INF)      # full-box alpha-CROWN (symmetric)
    D_zono = torch.full((N, N), INF)     # routed-zono (i->j over i's region)
    D_emp = torch.full((N, N), INF)      # empirical over i's routed region
    print("\nComputing bounds (full-box alphaCROWN, routed-zono) per ordered pair...")
    for i in range(N):
        routed_i = H[top1 == i]
        if routed_i.shape[0] < 8:
            continue
        k = min(16, routed_i.shape[0] - 1)
        center, V, acoef = zonotope_from_data(routed_i, n_components=k, margin=MARGIN)
        for j in range(N):
            if j == i:
                continue
            ei, ej = moe.experts[i], moe.experts[j]
            if not torch.isfinite(D_box[i, j]):
                b = pair_diff_bound(ei, ej, lo_full, hi_full,
                                    method="alpha_crown", alpha_iters=20)
                D_box[i, j] = D_box[j, i] = float(b)
            D_zono[i, j] = float(alpha_crown_zonotope_bound(
                ei, ej, center, V, acoef, n_iters=20))
            D_emp[i, j] = float(empirical_pair_diff(ei, ej, routed_i))

    print("\n  per-expert nearest twin: full-box vs routed-zono bound vs empirical")
    print(f"  {'expert':>7s} {'twin':>5s} {'box':>10s} {'zono':>9s} "
          f"{'emp(routed)':>11s} {'zono/emp':>9s}")
    for i in range(N):
        row = [(D_zono[i, j].item(), j) for j in range(N)
               if j != i and torch.isfinite(D_zono[i, j])]
        if not row:
            continue
        dz, j = min(row)
        de = D_emp[i, j].item()
        db = D_box[i, j].item()
        print(f"  {('E'+str(i)):>7s} {('E'+str(j)):>5s} {db:>10.1f} {dz:>9.2f} "
              f"{de:>11.3f} {dz/max(de,1e-6):>8.1f}x")

    # looseness summary
    def med(xs):
        xs = sorted(x for x in xs if x == x)
        return xs[len(xs)//2] if xs else float("nan")
    pairs = [(i, j) for i in range(N) for j in range(N)
             if j != i and torch.isfinite(D_zono[i, j])]
    loose_box = [D_box[i, j].item() / max(D_emp[i, j].item(), 1e-6) for i, j in pairs]
    loose_zono = [D_zono[i, j].item() / max(D_emp[i, j].item(), 1e-6) for i, j in pairs]
    print(f"\n  median looseness (bound/empirical):  full-box {med(loose_box):.1f}x"
          f"  ->  routed-zono {med(loose_zono):.1f}x")

    # budget curve under routed-zono bound
    print(f"\n{'='*60}\n  Aliasing budget curve (routed-zono certified)\n{'='*60}")
    finite = sorted(set(D_zono[i, j].item() for i, j in pairs))
    print(f"  {'delta_tol':>10s} {'#alias':>7s}  experts")
    for dt in finite[:5] + [finite[len(finite)//2], finite[-1]]:
        rep_of, aliased = aliasing_plan(D_zono, freq, dt)
        print(f"  {dt:>10.2f} {len(aliased):>7d}  {aliased}")

    # operating point: >=2 aliased
    chosen = None
    for dt in finite:
        rep_of, aliased = aliasing_plan(D_zono, freq, dt)
        if len(aliased) >= 2:
            chosen = (dt, rep_of, aliased); break
    if chosen is None:
        dt = finite[-1]; rep_of, aliased = aliasing_plan(D_zono, freq, dt)
        chosen = (dt, rep_of, aliased)
    dt, rep_of, aliased = chosen
    C = len(aliased)
    cert = max(D_zono[i, rep_of[i]].item() for i in aliased)
    drop = torch.zeros(N, dtype=torch.bool); drop[freq.argsort()[:C]] = True
    print(f"\n  operating point: alias {C} experts {aliased} -> "
          f"{[rep_of[i] for i in aliased]}")
    print(f"  CERTIFIED routed-zono bound <= {cert:.2f}")
    with torch.no_grad():
        y0 = switch_forward(moe, H)
        dev_a = (y0 - switch_forward(moe, H, rep_of=rep_of)).abs().max(-1).values.max().item()
        dev_f = (y0 - switch_forward(moe, H, drop_mask=drop)).abs().max(-1).values.max().item()
    torch.manual_seed(0)
    starts = torch.randperm(H.shape[0])[:12]
    pa = max(pgd_dev(moe, H[i:i+1], EPS, rep_of=rep_of) for i in starts)
    pf = max(pgd_dev(moe, H[i:i+1], EPS, drop_mask=drop) for i in starts)
    print(f"  clean worst dev:   aliasing {dev_a:.3f} | frequency {dev_f:.3f}")
    print(f"  eps-PGD worst dev: aliasing {pa:.3f} (cert {cert:.2f}) | "
          f"frequency {pf:.3f} (no cert)")

    print(f"\n{'='*60}")
    print("  KEY: if routed-zono looseness << full-box AND eps-PGD aliasing dev")
    print("  <= certified bound, the certificate is now a MEANINGFUL robustness")
    print("  guarantee that frequency pruning cannot provide.")


if __name__ == "__main__":
    main()
