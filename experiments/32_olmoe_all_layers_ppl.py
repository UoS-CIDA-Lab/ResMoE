"""Experiment 32 — Merge ALL 16 OLMoE layers with max_logit: does +5.1%/layer
compound or stay bounded?

Exp 31 showed merging layer 0 alone (64->32, cosine + max_logit router) costs
only +5.1% PPL vs lse_weight's +58.7%. The decisive test of whether the
improved merge is a usable compression method: merge EVERY layer (16 of 16)
and see if the per-layer cost compounds catastrophically or stays bounded.

Per layer we collect its own MLP-input calibration hidden states from the live
model (WikiText train, disjoint from the test set), build a cosine clustering
+ routing freq, and install a MergedOlmoeMoE (max_logit / lse_weight). Then we
score WikiText-2 test PPL with all layers merged.

Run: python3 experiments/32_olmoe_all_layers_ppl.py
"""
from __future__ import annotations

import sys
import pathlib
import math
import gc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cert_moe.merge import MergePlan, _build_label_of

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
TARGET = 32


def cluster_to_target(dist, target_n):
    """Complete-linkage agglomerative clustering down to exactly target_n
    clusters, single pass via the Lance-Williams update
    d(a∪b, k) = max(d(a,k), d(b,k)). Equivalent partition to
    complete_linkage_clusters at the cutoff that first yields target_n
    clusters, but O(N^2) per merge instead of re-scanning member pairs."""
    n = dist.shape[0]
    D = dist.clone().float()
    D.fill_diagonal_(math.inf)
    clusters = [[i] for i in range(n)]
    while len(clusters) > target_n:
        idx = int(D.argmin().item())
        a, b = divmod(idx, D.shape[0])
        if a > b:
            a, b = b, a
        newrow = torch.maximum(D[a], D[b])
        D[a] = newrow
        D[:, a] = newrow
        D[a, a] = math.inf
        keep = [i for i in range(D.shape[0]) if i != b]
        D = D[keep][:, keep]
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return MergePlan(clusters=clusters,
                     label_of=_build_label_of(clusters, n))


def cosine_distance(block):
    # Build on CPU to avoid a ~1.5GB GPU spike ([64, ~6.3M]); the live model
    # already occupies most of the 24GB card.
    with torch.no_grad():
        vecs = torch.stack([
            torch.cat([e.gate_proj.weight.flatten().float().cpu(),
                       e.up_proj.weight.flatten().float().cpu(),
                       e.down_proj.weight.flatten().float().cpu()])
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


class MergedOlmoeMoE(nn.Module):
    def __init__(self, block, plan, freq, mlp_config, scheme,
                 norm_topk_prob, device, dtype):
        super().__init__()
        from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
        self.scheme = scheme
        self.norm_topk_prob = norm_topk_prob
        self.num_experts = len(plan.clusters)
        self.top_k = min(block.top_k, self.num_experts)
        self.members = [torch.tensor(m, device=device) for m in plan.clusters]
        self.orig_router = block.gate.weight.detach().clone().to(device)
        with torch.no_grad():
            rows = torch.stack([
                torch.logsumexp(block.gate.weight[m].float(), dim=0)
                for m in plan.clusters
            ], dim=0).to(device=device, dtype=self.orig_router.dtype)
        self.lse_weight_rows = rows
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
        orig = hidden @ self.orig_router.T
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
        cl = self._cluster_logits(hidden_states)
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
def collect_all_layers(model, ids, device, n_layers, n_tokens=2048):
    """Per-layer MLP-input hidden states from one forward pass."""
    caps = {li: [] for li in range(n_layers)}
    handles = []
    for li in range(n_layers):
        mlp = model.model.layers[li].mlp
        def mk(li):
            def hook(_m, args):
                caps[li].append(args[0].detach().reshape(
                    -1, args[0].shape[-1]).float().cpu())
            return hook
        handles.append(mlp.register_forward_pre_hook(mk(li)))
    try:
        inp = ids[:n_tokens].unsqueeze(0).to(device)
        model(inp)
    finally:
        for h in handles:
            h.remove()
    return {li: torch.cat(caps[li]) for li in range(n_layers)}


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


def load_model(device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    return m.to(device).eval()


def merge_all(model, calib, device, scheme):
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        block = model.model.layers[li].mlp
        K, N = block.top_k, block.num_experts
        mlp_config = block.experts[0].config
        norm_topk = getattr(block, "norm_topk_prob", False)
        dtype = block.gate.weight.dtype
        H = calib[li]
        router_w = block.gate.weight.detach().float().cpu()
        freq = routing_freq(H, router_w, K, N)
        d_cos = cosine_distance(block).cpu()
        plan = cluster_to_target(d_cos, TARGET)
        merged = MergedOlmoeMoE(block, plan, freq, mlp_config, scheme,
                                norm_topk, device, dtype)
        model.model.layers[li].mlp = merged
        del block
        gc.collect(); torch.cuda.empty_cache()
    return model


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 (train calib + test eval)...")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train_text = "\n\n".join(t for t in train["text"] if t.strip())
    test_text = "\n\n".join(t for t in test["text"] if t.strip())
    calib_ids = tok(train_text, return_tensors="pt").input_ids[0]
    test_ids = tok(test_text, return_tensors="pt").input_ids[0]
    MAX_WINDOWS = 40

    print(f"\nLoading {MODEL_NAME} for baseline + calibration...")
    model = load_model(device)
    n_layers = len(model.model.layers)
    print(f"  {n_layers} layers, each MoE N={model.model.layers[0].mlp.num_experts}")

    print("Collecting per-layer calibration hidden states...")
    calib = collect_all_layers(model, calib_ids, device, n_layers)
    print(f"  per-layer calib tokens: {calib[0].shape[0]}")

    print("Baseline PPL...")
    ppl_base, ntok = compute_ppl(model, test_ids, device, max_windows=MAX_WINDOWS)
    print(f"  baseline PPL = {ppl_base:.4f}  ({ntok} tokens)")
    del model
    gc.collect(); torch.cuda.empty_cache()

    results = {"baseline": ppl_base}
    for scheme in ("max_logit", "lse_weight"):
        print(f"\nLoading fresh model; merging ALL {n_layers} layers "
              f"64->{TARGET} with {scheme}...")
        model = load_model(device)
        model = merge_all(model, calib, device, scheme)
        ppl, _ = compute_ppl(model, test_ids, device, max_windows=MAX_WINDOWS)
        results[scheme] = ppl
        d = (ppl - ppl_base) / ppl_base * 100
        print(f"  ALL-LAYERS {scheme:11s} PPL {ppl:.4f}  (Δ {d:+.2f}%)")
        del model
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*64}\n  SUMMARY — all 16 layers merged 64->{TARGET}\n{'='*64}")
    print(f"  baseline                 PPL {results['baseline']:8.4f}")
    for scheme in ("max_logit", "lse_weight"):
        d = (results[scheme] - results['baseline']) / results['baseline'] * 100
        print(f"  ALL-LAYERS {scheme:11s}   PPL {results[scheme]:8.4f}  "
              f"(Δ {d:+.2f}%)")
    print(f"\n  exp31 (layer 0 ONLY, max_logit): +5.09%.")
    print("  If all-16 max_logit stays modest (not ~16x), the improved merge")
    print("  is a usable end-to-end compressor; if it explodes, per-layer")
    print("  errors compound through the residual stream.")


if __name__ == "__main__":
    main()
