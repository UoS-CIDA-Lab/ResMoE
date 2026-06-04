"""Experiment 36 — Does HC-SMoE's aliasing router (keep the FULL router) fix
OLMoE merging where our router-reduction degrades it?

HC-SMoE (arXiv 2410.08589, references/2410.08589v4.pdf) keeps the original
n-way router unchanged and only ALIASES each selected expert to its merged
group (§3.1) — identical top-K decisions to the original model, zero routing
error. Our framework instead REDUCES the router to r clusters and re-runs
top-K, which exp 27/31 showed is the dominant error.

This isolates the router-handling effect: for the SAME clustering, compare
  - reduce : cluster-reduced router + max_logit (our best operator, exp 31)
  - alias  : keep full 64-way router, remap selected experts to merged group
on OLMoE all-16-layers at 64->48 (25%) and 64->32 (50%, the paper's headline),
metric = AG-News zero-shot accuracy + WikiText PPL, 2048-tok/layer calib.

Clusterings: cosine, and HC-SMoE's own (per-expert mean-output vector +
AVERAGE linkage + Euclidean — the combo the paper reports as best).
Prediction: aliasing holds up far better, especially at 50%, confirming
router reduction was the culprit (and matching HC-SMoE's robustness).

Run: python3 experiments/36_olmoe_aliasing_router.py
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
AGNEWS_LABELS = [" World", " Sports", " Business", " Technology"]
AGNEWS_PROMPT = "News: {text}\nTopic:"


def cosine_distance(block):
    with torch.no_grad():
        vecs = torch.stack([
            torch.cat([e.gate_proj.weight.flatten().float().cpu(),
                       e.up_proj.weight.flatten().float().cpu(),
                       e.down_proj.weight.flatten().float().cpu()])
            for e in block.experts])
    vecs = F.normalize(vecs, dim=-1)
    d = 1.0 - vecs @ vecs.T
    d.fill_diagonal_(0.0)
    return d


@torch.no_grad()
def hcsmoe_distance(block, H, device, cap=2048):
    """HC-SMoE similarity: Euclidean distance between per-expert MEAN OUTPUT
    vectors o_j = mean_x E_j(x) over calibration tokens."""
    if H.shape[0] > cap:
        H = H[torch.randperm(H.shape[0])[:cap]]
    Hd = H.to(device=device, dtype=block.gate.weight.dtype)
    o = torch.stack([block.experts[i](Hd).float().mean(0)
                     for i in range(block.num_experts)])   # [N, d]
    d = torch.cdist(o, o).cpu()                              # [N, N]
    d.fill_diagonal_(0.0)
    return d


def routing_freq(H, router_weight, K, n_experts):
    logits = H @ router_weight.T
    _, idx = logits.topk(K, dim=-1)
    freq = torch.zeros(n_experts)
    for k in range(K):
        freq.scatter_add_(0, idx[:, k], torch.ones(H.shape[0]))
    return freq


def cluster_complete(dist, target_n):
    return _agglomerate(dist, target_n, linkage="complete")


def cluster_average(dist, target_n):
    return _agglomerate(dist, target_n, linkage="average")


def _agglomerate(dist, target_n, linkage):
    """Single-pass Lance-Williams agglomerative clustering."""
    n0 = dist.shape[0]
    D = dist.clone().float()
    D.fill_diagonal_(math.inf)
    clusters = [[i] for i in range(n0)]
    sizes = [1] * n0
    while len(clusters) > target_n:
        idx = int(D.argmin().item())
        a, b = divmod(idx, D.shape[0])
        if a > b:
            a, b = b, a
        if linkage == "complete":
            nr = torch.maximum(D[a], D[b])
        else:  # average
            sa, sb = sizes[a], sizes[b]
            nr = (sa * D[a] + sb * D[b]) / (sa + sb)
        D[a] = nr; D[:, a] = nr; D[a, a] = math.inf
        keep = [i for i in range(D.shape[0]) if i != b]
        D = D[keep][:, keep]
        clusters[a] = clusters[a] + clusters[b]
        sizes[a] = sizes[a] + sizes[b]
        del clusters[b]; del sizes[b]
    return MergePlan(clusters=clusters, label_of=_build_label_of(clusters, n0))


def build_merged_experts(block, plan, freq, mlp_config, device, dtype):
    from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
    experts = nn.ModuleList()
    with torch.no_grad():
        for members in plan.clusters:
            w = freq[members].float()
            w = w / w.sum() if w.sum() > 0 else torch.ones_like(w) / len(w)
            e = OlmoeMLP(mlp_config).to(device=device, dtype=dtype)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                acc = sum(w[i] * getattr(block.experts[m], proj).weight.float()
                          for i, m in enumerate(members))
                getattr(e, proj).weight.copy_(acc.to(dtype))
            experts.append(e)
    return experts


class ReduceRouterMoE(nn.Module):
    """Our scheme: reduce router to r clusters, re-run top-K (max_logit)."""
    def __init__(self, block, plan, experts, device):
        super().__init__()
        self.norm_topk_prob = getattr(block, "norm_topk_prob", False)
        self.num_experts = len(plan.clusters)
        self.top_k = min(block.top_k, self.num_experts)
        self.members = [torch.tensor(m, device=device) for m in plan.clusters]
        self.orig_router = block.gate.weight.detach().clone().to(device)
        self.experts = experts

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        orig = x @ self.orig_router.T
        cl = torch.stack([orig.index_select(1, m).max(dim=1).values
                          for m in self.members], dim=1)
        routing = F.softmax(cl, dim=1, dtype=torch.float)
        routing, sel = torch.topk(routing, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing = routing / routing.sum(dim=-1, keepdim=True)
        routing = routing.to(x.dtype)
        final = torch.zeros((b * s, h), dtype=x.dtype, device=x.device)
        mask = F.one_hot(sel, self.num_experts).permute(2, 1, 0)
        for c in range(self.num_experts):
            i, tx = torch.where(mask[c])
            if tx.numel():
                final.index_add_(0, tx,
                                 (self.experts[c](x[tx]) * routing[tx, i, None]).to(final.dtype))
        return final.view(b, s, h), cl


class AliasRouterMoE(nn.Module):
    """HC-SMoE scheme: keep the FULL n-way router, original top-K selection,
    alias each selected expert to its merged group (summing gate weights)."""
    def __init__(self, block, plan, experts, device):
        super().__init__()
        self.norm_topk_prob = getattr(block, "norm_topk_prob", False)
        self.K = block.top_k
        self.N = block.num_experts
        self.r = len(plan.clusters)
        self.router = block.gate.weight.detach().clone().to(device)
        self.label_of = torch.tensor(plan.label_of, device=device)
        self.experts = experts

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        T = x.shape[0]
        logits = x @ self.router.T                       # [T, N] full router
        probs = F.softmax(logits, dim=1, dtype=torch.float)
        topv, topi = probs.topk(self.K, dim=1)           # original top-K
        if self.norm_topk_prob:
            topv = topv / topv.sum(dim=1, keepdim=True)
        # aggregate gate weight into the merged group of each selected expert
        groups = self.label_of[topi]                     # [T, K]
        gw = torch.zeros(T, self.r, dtype=torch.float, device=x.device)
        gw.scatter_add_(1, groups, topv)
        gw = gw.to(x.dtype)
        final = torch.zeros((T, h), dtype=x.dtype, device=x.device)
        for g in range(self.r):
            sel = gw[:, g] > 0
            if sel.any():
                final[sel] += gw[sel, g, None] * self.experts[g](x[sel])
        return final.view(b, s, h), logits


@torch.no_grad()
def collect_all_layers(model, ids, device, n_layers, n_tokens=2048):
    caps = {li: [] for li in range(n_layers)}
    handles = []
    for li in range(n_layers):
        def mk(li):
            def hook(_m, args):
                caps[li].append(args[0].detach().reshape(
                    -1, args[0].shape[-1]).float().cpu())
            return hook
        handles.append(
            model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    try:
        model(ids[:n_tokens].unsqueeze(0).to(device))
    finally:
        for h in handles:
            h.remove()
    return {li: torch.cat(caps[li]) for li in range(n_layers)}


@torch.no_grad()
def compute_ppl(model, ids, device, ctx=1024, max_windows=30):
    nll, ntok, nw = 0.0, 0, 0
    for begin in range(0, ids.shape[0] - 1, ctx):
        inp = ids[begin:min(begin + ctx, ids.shape[0])].unsqueeze(0).to(device)
        tgt = inp.clone(); tgt[:, :-1] = inp[:, 1:]; tgt[:, -1] = -100
        logits = model(inp).logits.float()
        sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
        st = tgt[:, :-1].reshape(-1)
        nll += F.cross_entropy(sl, st, reduction="sum").item()
        ntok += (st != -100).sum().item(); nw += 1
        if nw >= max_windows:
            break
    return math.exp(nll / max(ntok, 1))


@torch.no_grad()
def agnews_acc(model, tok, examples, device, batch=48):
    lab_ids = [tok(l, add_special_tokens=False).input_ids for l in AGNEWS_LABELS]
    seqs, meta = [], []
    for ei, (text, gold) in enumerate(examples):
        p = tok(AGNEWS_PROMPT.format(text=text)).input_ids
        for lab in lab_ids:
            seqs.append(p + lab); meta.append((len(p), len(lab)))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    lp = [0.0] * len(seqs)
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        ml = max(len(x) for x in chunk)
        inp = torch.full((len(chunk), ml), pad_id, dtype=torch.long)
        am = torch.zeros((len(chunk), ml), dtype=torch.long)
        for i, x in enumerate(chunk):
            inp[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
        logp = F.log_softmax(
            model(inp.to(device), attention_mask=am.to(device)).logits.float(), dim=-1)
        for i in range(len(chunk)):
            plen, llen = meta[s + i]
            lp[s + i] = sum(logp[i, plen + k - 1, chunk[i][plen + k]].item()
                            for k in range(llen)) / llen
    correct = 0
    for ei, (_, gold) in enumerate(examples):
        sc = [lp[ei * 4 + c] for c in range(4)]
        correct += int(max(range(4), key=lambda c: sc[c]) == gold)
    return correct / len(examples)


def load_model(device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    return m.to(device).eval()


def merge_all(model, calib, device, clustering, router_mode, ratio):
    for li in range(len(model.model.layers)):
        block = model.model.layers[li].mlp
        cfg = block.experts[0].config
        dtype = block.gate.weight.dtype
        rw = block.gate.weight.detach().float().cpu()
        freq = routing_freq(calib[li], rw, block.top_k, block.num_experts)
        if clustering == "cosine":
            plan = cluster_complete(cosine_distance(block).cpu(), ratio)
        elif clustering == "hcsmoe":
            plan = cluster_average(hcsmoe_distance(block, calib[li], device), ratio)
        else:
            raise ValueError(clustering)
        experts = build_merged_experts(block, plan, freq, cfg, device, dtype)
        if router_mode == "reduce":
            merged = ReduceRouterMoE(block, plan, experts, device)
        else:
            merged = AliasRouterMoE(block, plan, experts, device)
        model.model.layers[li].mlp = merged
        del block
        gc.collect(); torch.cuda.empty_cache()
    return model


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Hard-cap GPU memory at 75% of total (user constraint).
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"GPU memory capped at 75% of {tot:.0f} MiB = {tot*0.75:.0f} MiB")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_ids = tok("\n\n".join(t for t in test["text"] if t.strip()),
                   return_tensors="pt").input_ids[0]
    calib_ids = tok("\n\n".join(t for t in train["text"] if t.strip()),
                    return_tensors="pt").input_ids[0]
    ag = load_dataset("ag_news", split="test")
    ex = [(ag[i]["text"], ag[i]["label"]) for i in range(500)]

    print(f"Loading {MODEL_NAME}...")
    model = load_model(device)
    n_layers = len(model.model.layers)
    calib = collect_all_layers(model, calib_ids, device, n_layers, 2048)
    ppl0 = compute_ppl(model, test_ids, device)
    acc0 = agnews_acc(model, tok, ex, device)
    print(f"  baseline: PPL {ppl0:.2f}  AG-acc {acc0*100:.1f}%")
    del model; gc.collect(); torch.cuda.empty_cache()

    for ratio, pct in ((48, 25), (32, 50)):
        print(f"\n{'='*70}\n  all-16-layers 64->{ratio} ({pct}% reduction)"
              f"   AG-acc | PPL\n{'='*70}")
        print(f"  {'clustering':12s} {'router':8s} {'AG-acc':>8s} {'PPL':>9s}")
        print("  " + "-" * 42)
        for clustering in ("cosine", "hcsmoe"):
            for rmode in ("reduce", "alias"):
                model = load_model(device)
                model = merge_all(model, calib, device, clustering, rmode, ratio)
                ppl = compute_ppl(model, test_ids, device)
                acc = agnews_acc(model, tok, ex, device)
                print(f"  {clustering:12s} {rmode:8s} {acc*100:>7.1f}% "
                      f"{ppl:>9.1f}")
                del model; gc.collect(); torch.cuda.empty_cache()
        print(f"  {'(baseline)':12s} {'--':8s} {acc0*100:>7.1f}% {ppl0:>9.1f}")

    print(f"\n{'='*70}")
    print("  READ: if 'alias' >> 'reduce' on accuracy (esp. at 50%), the")
    print("  router REDUCTION was the culprit; keeping the full router")
    print("  (HC-SMoE) is what preserves quality. cosine vs hcsmoe isolates")
    print("  the clustering effect under the good router.")


if __name__ == "__main__":
    main()
