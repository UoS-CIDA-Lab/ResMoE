"""Experiment 31 — Integrated fixes on OLMoE top-8: does REAL PPL improve?

Exp 26-30 rehabilitated bound-driven merging on top-1 Switch via three fixes:
(1) better ranking, (2) max_logit router merge, (3) protect-top-traffic
frequency-awareness. Exp 18 measured real WikiText-2 PPL after merging OLMoE
layer 0 (64->32) with cosine + lse_weight router = +58.7% PPL (18.24 -> 28.95);
cert-aware was far worse. Those used apply_merge's lse_weight (the WORST router
merge, exp 27). This experiment installs the IMPROVED recipe into the real
OLMoE-1B-7B and re-measures PPL.

For OLMoE SwiGLU the discriminative bound is unavailable (swiglu_bounds are
~10000x loose / non-discriminative, see olmoe-port-status), so clustering uses
COSINE; the transferable fixes are the ROUTER MERGE (lse_logit / max_logit vs
the default lse_weight) and PROTECT-TOP-T frequency-awareness.

Because max_logit / lse_logit are nonlinear in the per-expert logits (not a
fixed linear gate), we replace layer-0's MoE forward with a module that keeps
the original 64-row router, forms cluster logits per scheme, and routes top-K
clusters with faithful OLMoE gating (norm_topk_prob honored).

Run: python3 experiments/31_olmoe_integrated_ppl.py
"""
from __future__ import annotations

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cert_moe.merge import complete_linkage_clusters, MergePlan, _build_label_of

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
LAYER_IDX = 0
CACHE = pathlib.Path(__file__).resolve().parents[1] / "olmoe_layer0_cache.pt"


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


def find_cutoff(dist, target_n):
    flat = sorted(set(d for d in dist.flatten().tolist() if math.isfinite(d)))
    for eps in flat:
        if len(complete_linkage_clusters(dist, epsilon=eps).clusters) <= target_n:
            return eps
    return flat[-1] if flat else 1.0


def plan_of(clusters, N):
    return MergePlan(clusters=clusters, label_of=_build_label_of(clusters, N))


def cluster_protect(dist, freq, target_n, T, N):
    """Force top-T traffic experts to singletons; cluster the rest."""
    prot = freq.argsort(descending=True)[:T].tolist()
    rest = [i for i in range(N) if i not in prot]
    sub = dist[rest][:, rest].clone()
    subplan = complete_linkage_clusters(sub, find_cutoff(sub, target_n - T))
    clusters = [[rest[m] for m in members] for members in subplan.clusters]
    clusters += [[p] for p in prot]
    return plan_of(clusters, N)


class MergedOlmoeMoE(nn.Module):
    """Drop-in replacement for OlmoeSparseMoeBlock with a cluster router-merge
    scheme. Keeps the original 64-row router to form per-expert logits, derives
    cluster logits per `scheme`, routes top-K clusters with OLMoE gating."""

    def __init__(self, block, plan, freq, mlp_config, scheme,
                 norm_topk_prob, device, dtype):
        super().__init__()
        from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
        self.scheme = scheme
        self.norm_topk_prob = norm_topk_prob
        self.num_experts = len(plan.clusters)
        self.top_k = min(block.top_k, self.num_experts)
        self.members = [torch.tensor(m, device=device) for m in plan.clusters]
        # original 64-row router (frozen)
        self.orig_router = block.gate.weight.detach().clone().to(device)
        # precomputed lse_weight rows (for the lse_weight scheme)
        with torch.no_grad():
            rows = torch.stack([
                torch.logsumexp(block.gate.weight[m].float(), dim=0)
                for m in plan.clusters
            ], dim=0).to(device=device, dtype=self.orig_router.dtype)
        self.lse_weight_rows = rows
        # merged cluster experts (freq-weighted average)
        self.experts = nn.ModuleList()
        with torch.no_grad():
            for members in plan.clusters:
                w = freq[members].float()
                w = w / w.sum() if w.sum() > 0 else torch.ones_like(w) / len(w)
                e = OlmoeMLP(mlp_config).to(device=device, dtype=dtype)
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    acc = sum(w[i] * getattr(block.experts[m], proj).weight.float()
                              for i, m in enumerate(members))
                    getattr(e, proj).weight.copy_(acc.to(dtype))
                self.experts.append(e)

    def _cluster_logits(self, hidden):
        if self.scheme == "lse_weight":
            return hidden @ self.lse_weight_rows.T
        orig = hidden @ self.orig_router.T          # [T, 64]
        cols = []
        for m in self.members:
            sub = orig.index_select(1, m)
            if self.scheme == "lse_logit":
                cols.append(torch.logsumexp(sub.float(), dim=1).to(orig.dtype))
            elif self.scheme == "max_logit":
                cols.append(sub.max(dim=1).values)
            else:
                raise ValueError(self.scheme)
        return torch.stack(cols, dim=1)

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        cl = self._cluster_logits(hidden_states)               # [T, N']
        routing = F.softmax(cl, dim=1, dtype=torch.float)
        routing, selected = torch.topk(routing, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing = routing / routing.sum(dim=-1, keepdim=True)
        routing = routing.to(hidden_states.dtype)
        final = torch.zeros((b * s, h), dtype=hidden_states.dtype,
                            device=hidden_states.device)
        mask = F.one_hot(selected, self.num_experts).permute(2, 1, 0)
        for c in range(self.num_experts):
            idx, top_x = torch.where(mask[c])
            if top_x.numel() == 0:
                continue
            cur = self.experts[c](hidden_states[top_x])
            cur = cur * routing[top_x, idx, None]
            final.index_add_(0, top_x, cur.to(final.dtype))
        return final.view(b, s, h), cl


@torch.no_grad()
def compute_ppl(model, ids, device, ctx=1024, stride=1024, max_windows=None):
    nll_sum, n_tok, n_win = 0.0, 0, 0
    for begin in range(0, ids.shape[0] - 1, stride):
        end = min(begin + ctx, ids.shape[0])
        inp = ids[begin:end].unsqueeze(0).to(device)
        target = inp.clone()
        target[:, :-1] = inp[:, 1:]
        target[:, -1] = -100
        logits = model(inp).logits.float()
        sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
        st = target[:, :-1].reshape(-1)
        nll_sum += F.cross_entropy(sl, st, reduction="sum").item()
        n_tok += (st != -100).sum().item()
        n_win += 1
        if max_windows and n_win >= max_windows:
            break
    return math.exp(nll_sum / max(n_tok, 1)), n_tok


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model.to(device).eval()

    layer = model.model.layers[LAYER_IDX]
    block = layer.mlp
    K, N = block.top_k, block.num_experts
    d_ff = block.experts[0].gate_proj.weight.shape[0]
    mlp_config = block.experts[0].config
    norm_topk = getattr(block, "norm_topk_prob",
                        getattr(model.config, "norm_topk_prob", False))
    dtype = block.gate.weight.dtype
    print(f"Layer {LAYER_IDX}: N={N}, K={K}, norm_topk_prob={norm_topk}")

    cache = torch.load(CACHE, weights_only=True, map_location="cpu")
    H = cache["H"].float(); del cache
    router_w = block.gate.weight.detach().float().cpu()
    freq = routing_freq(H, router_w, K, N)
    print(f"Calib H={tuple(H.shape)}; top freq experts: "
          f"{freq.argsort(descending=True)[:6].tolist()} "
          f"(max {int(freq.max())}, min {int(freq.min())})")
    d_cos = cosine_distance(block).cpu()

    print("Loading WikiText-2 test...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    MAX_WINDOWS = 40

    print("\nBaseline PPL...")
    ppl_base, ntok = compute_ppl(model, ids, device, max_windows=MAX_WINDOWS)
    print(f"  baseline PPL = {ppl_base:.4f}  ({ntok} tokens)")

    TARGET = 32
    eps = find_cutoff(d_cos, TARGET)
    plan_cos = complete_linkage_clusters(d_cos, epsilon=eps)

    # recipes: (tag, plan, scheme)
    recipes = [
        ("cosine + lse_weight (exp18)", plan_cos, "lse_weight"),
        ("cosine + lse_logit",          plan_cos, "lse_logit"),
        ("cosine + max_logit",          plan_cos, "max_logit"),
        ("cosine + max_logit + protect1", cluster_protect(d_cos, freq, TARGET, 1, N), "max_logit"),
        ("cosine + max_logit + protect4", cluster_protect(d_cos, freq, TARGET, 4, N), "max_logit"),
        ("cosine + lse_logit + protect4", cluster_protect(d_cos, freq, TARGET, 4, N), "lse_logit"),
    ]

    print(f"\n{'='*70}\n  OLMoE layer {LAYER_IDX} merged 64 -> {TARGET}, "
          f"WikiText-2 PPL\n{'='*70}")
    results = {"baseline": ppl_base}
    for tag, plan, scheme in recipes:
        merged = MergedOlmoeMoE(block, plan, freq, mlp_config, scheme,
                                norm_topk, device, dtype)
        layer.mlp = merged
        ppl, _ = compute_ppl(model, ids, device, max_windows=MAX_WINDOWS)
        layer.mlp = block
        results[tag] = ppl
        d = (ppl - ppl_base) / ppl_base * 100
        print(f"  {tag:32s} PPL {ppl:8.4f}  (Δ {d:+.2f}%)")

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  baseline                         PPL {ppl_base:8.4f}")
    for tag, _, _ in recipes:
        d = (results[tag] - ppl_base) / ppl_base * 100
        print(f"  {tag:32s} PPL {results[tag]:8.4f}  (Δ {d:+.2f}%)")
    print("\n  exp18 reference: cosine+lse_weight 64->32 = +58.7% (28.95).")
    print("  Better router merge + protect-top-T should lower these Δ's.")


if __name__ == "__main__":
    main()
