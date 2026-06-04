"""Experiment 20 — Does certification predict per-token quality?

THE decisive test for whether cert% means anything. Experiment 19 showed
cert-aware merging never beats cosine on aggregate perplexity, but retains
a certified fraction (e.g. 28.6% at bound_weight=0.5) that cosine cannot.
That coverage is only valuable if certified tokens are actually SAFER —
i.e. the merge degrades them less than uncertified tokens.

We build the bound_weight=0.5 cert-aware merge on OLMoE layer 0 (64→32),
then for every WikiText-2 eval token compute:
  - per-token NLL under the ORIGINAL model
  - per-token NLL under the MERGED model
  - degradation = NLL_merged - NLL_orig
  - certified? (Theorem-1 conditions on that token's layer-0 MoE input)

If certified tokens have much smaller degradation, certification is a real
safety filter. If not, cert% is decorative and cosine dominates outright.

Needs HF_HUB_DISABLE_XET=1 to (down)load the model.
Run: HF_HUB_DISABLE_XET=1 python3 experiments/20_olmoe_cert_predicts_quality.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cert_moe.toy_swiglu_moe import ToySwiGLUMoEConfig
from cert_moe.merge import cert_aware_greedy_merge
from cert_moe.theorem1_conditions import check_conditions

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
LAYER_IDX = 0
TARGET_N = 32
BOUND_WEIGHT = 0.5
MAX_WINDOWS = 40
CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


class LiteMoE:
    def __init__(self, router_weight, n_experts, top_k, d_model, d_ff):
        self.cfg = ToySwiGLUMoEConfig(
            d_model=d_model, d_ff=d_ff, n_experts=n_experts,
            top_k=top_k, n_clone_pairs=0,
        )
        self.router = nn.Linear(d_model, n_experts, bias=False)
        with torch.no_grad():
            self.router.weight.copy_(router_weight)


def cosine_distance(block):
    vecs = torch.stack([
        torch.cat([e.gate_proj.weight.flatten(),
                   e.up_proj.weight.flatten(),
                   e.down_proj.weight.flatten()]).float()
        for e in block.experts
    ])
    vecs = F.normalize(vecs, dim=-1)
    dist = 1.0 - vecs @ vecs.T
    dist.fill_diagonal_(0.0)
    return dist


def routing_freq(H, router_weight, K, n_experts):
    logits = H @ router_weight.T
    _, idx = logits.topk(K, dim=-1)
    freq = torch.zeros(n_experts)
    for k in range(K):
        freq.scatter_add_(0, idx[:, k], torch.ones(H.shape[0]))
    return freq


def build_merged_modules(block, plan, freq, mlp_config, device, dtype):
    from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
    d = block.gate.weight.shape[1]
    Np = len(plan.clusters)
    new_gate = nn.Linear(d, Np, bias=False).to(device=device, dtype=dtype)
    new_experts = nn.ModuleList()
    with torch.no_grad():
        for c, members in enumerate(plan.clusters):
            w = freq[members].float()
            w = w / w.sum() if w.sum() > 0 else torch.ones_like(w) / len(w)
            e = OlmoeMLP(mlp_config).to(device=device, dtype=dtype)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                acc = sum(w[i] * getattr(block.experts[m], proj).weight.float()
                          for i, m in enumerate(members))
                getattr(e, proj).weight.copy_(acc.to(dtype))
            new_gate.weight[c].copy_(
                torch.logsumexp(block.gate.weight[members].float(), dim=0).to(dtype))
            new_experts.append(e)
    return new_gate, new_experts, Np


@torch.no_grad()
def per_token_nll(model, ids, device, ctx=1024, stride=1024, max_windows=None,
                  capture_block=None):
    """Yield, per window, (loss[L-1], hidden[L-1, d] or None)."""
    captured = {}
    handle = None
    if capture_block is not None:
        def pre_hook(_m, args):
            captured["h"] = args[0].detach()
        handle = capture_block.register_forward_pre_hook(pre_hook)
    out_loss, out_hidden = [], []
    try:
        n_win = 0
        for begin in range(0, ids.shape[0] - 1, stride):
            end = min(begin + ctx, ids.shape[0])
            inp = ids[begin:end].unsqueeze(0).to(device)
            logits = model(inp).logits.float()
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = inp[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
            out_loss.append(loss.cpu())
            if capture_block is not None:
                h = captured["h"].reshape(-1, captured["h"].shape[-1])
                out_hidden.append(h[:-1].float().cpu())  # align to positions 0..L-2
            n_win += 1
            if max_windows and n_win >= max_windows:
                break
    finally:
        if handle is not None:
            handle.remove()
    loss_all = torch.cat(out_loss)
    hidden_all = torch.cat(out_hidden) if out_hidden else None
    return loss_all, hidden_all


def ppl(nll_tensor):
    return math.exp(nll_tensor.mean().item()) if nll_tensor.numel() else float("nan")


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model.to(device).eval()

    block = model.model.layers[LAYER_IDX].mlp
    K_orig, N_orig = block.top_k, block.num_experts
    d_model = block.gate.weight.shape[1]
    d_ff = block.experts[0].gate_proj.weight.shape[0]
    mlp_config = block.experts[0].config
    dtype = block.gate.weight.dtype
    orig_gate, orig_experts = block.gate, block.experts

    cache = torch.load(CACHE, weights_only=True, map_location="cpu")
    H = cache["H"].float(); del cache
    router_w = block.gate.weight.detach().float().cpu()
    freq = routing_freq(H, router_w, K_orig, N_orig)
    d_cos = cosine_distance(block).cpu()
    lite = LiteMoE(router_w, N_orig, K_orig, d_model, d_ff)
    max_b = d_cos[d_cos.isfinite()].max().item() * 1.001

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]

    print("Pass 1: original per-token NLL...")
    nll_orig, _ = per_token_nll(model, ids, device, max_windows=MAX_WINDOWS)
    print(f"  original PPL = {ppl(nll_orig):.4f}  ({nll_orig.numel()} tokens)")

    print(f"\nBuilding cert-aware plan (bound_weight={BOUND_WEIGHT})...", flush=True)
    plan = cert_aware_greedy_merge(
        lite, H, d_cos, target_n_clusters=TARGET_N,
        max_bound=max_b, delta_em=0.1, bound_weight=BOUND_WEIGHT)

    g, ex, Np = build_merged_modules(block, plan, freq, mlp_config, device, dtype)
    block.gate, block.experts = g, ex
    block.num_experts, block.top_k = Np, min(K_orig, Np)

    print("Pass 2: merged per-token NLL + layer-0 MoE input capture...")
    nll_merged, hidden = per_token_nll(
        model, ids, device, max_windows=MAX_WINDOWS, capture_block=block)
    print(f"  merged PPL   = {ppl(nll_merged):.4f}")

    # restore
    block.gate, block.experts = orig_gate, orig_experts
    block.num_experts, block.top_k = N_orig, K_orig

    # Per-token certification on the captured layer-0 MoE inputs
    n = min(nll_orig.numel(), nll_merged.numel(), hidden.shape[0])
    nll_orig, nll_merged, hidden = nll_orig[:n], nll_merged[:n], hidden[:n]
    cert = check_conditions(hidden, router_w, plan, K=K_orig,
                            delta_threshold=0.1).all_certified
    degr = nll_merged - nll_orig

    c, u = cert, ~cert
    print(f"\n{'='*66}")
    print(f"  DECISIVE TEST — does certification predict per-token quality?")
    print(f"  OLMoE layer 0, 64→32, bound_weight={BOUND_WEIGHT}, "
          f"{n} tokens")
    print(f"{'='*66}")
    print(f"  certified tokens   : {c.sum().item():5d} ({c.float().mean()*100:.1f}%)")
    print(f"  uncertified tokens : {u.sum().item():5d} ({u.float().mean()*100:.1f}%)")
    print()
    print(f"  {'group':18s} {'orig PPL':>10s} {'merged PPL':>11s} "
          f"{'mean NLL degr':>14s}")
    print("  " + "-" * 56)
    for name, mask in (("certified", c), ("uncertified", u), ("all", torch.ones_like(c))):
        if mask.sum() == 0:
            continue
        print(f"  {name:18s} {ppl(nll_orig[mask]):>10.3f} "
              f"{ppl(nll_merged[mask]):>11.3f} {degr[mask].mean().item():>14.4f}")
    print()
    if c.sum() > 0 and u.sum() > 0:
        ratio = degr[u].mean().item() / max(degr[c].mean().item(), 1e-6)
        print(f"  degradation ratio (uncertified / certified) = {ratio:.2f}x")
        print(f"  → >1 means certification DOES identify safer tokens.")


if __name__ == "__main__":
    main()
