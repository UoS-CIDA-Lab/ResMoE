"""Experiment 34 (probe) — pick a usable all-layers compression ratio, and
validate AG-News zero-shot accuracy as a real downstream metric.

Goal: upgrade the baseline comparison toward the way MC-SMoE / HC-SMoE
actually evaluate (full model, downstream zero-shot accuracy, larger
calibration) instead of single-layer rel_div on 32 tokens. Exp 32 showed
all-16-layers 64->32 explodes (PPL 1040); before running the full
method x operator grid we must find a ratio where the merged model is still
usable. This probe sweeps all-layers cosine + max_logit at 64->{56,48,40},
reporting WikiText-2 PPL and AG-News (4-way) zero-shot accuracy, with a
1024-token/layer calibration collected from the live model.

AG-News zero-shot: prompt + 4 label verbalizers, pick the label whose tokens
have the highest length-normalized log-likelihood (the lm-eval multiple-choice
protocol). Datasets are cached offline.

Run: python3 experiments/34_olmoe_ratio_probe.py
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


def cluster_to_target(dist, target_n):
    D = dist.clone().float()
    D.fill_diagonal_(math.inf)
    clusters = [[i] for i in range(dist.shape[0])]
    while len(clusters) > target_n:
        idx = int(D.argmin().item())
        a, b = divmod(idx, D.shape[0])
        if a > b:
            a, b = b, a
        nr = torch.maximum(D[a], D[b])
        D[a] = nr; D[:, a] = nr; D[a, a] = math.inf
        keep = [i for i in range(D.shape[0]) if i != b]
        D = D[keep][:, keep]
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return MergePlan(clusters=clusters,
                     label_of=_build_label_of(clusters, dist.shape[0]))


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
def collect_all_layers(model, ids, device, n_layers, n_tokens=1024):
    caps = {li: [] for li in range(n_layers)}
    handles = []
    for li in range(n_layers):
        def mk(li):
            def hook(_m, args):
                caps[li].append(args[0].detach().reshape(
                    -1, args[0].shape[-1]).float().cpu())
            return hook
        handles.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
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
def agnews_acc(model, tok, examples, device):
    """Length-normalized log-likelihood argmax over 4 label verbalizers."""
    label_ids = [tok(l, add_special_tokens=False).input_ids for l in AGNEWS_LABELS]
    correct = 0
    for text, gold in examples:
        prompt = AGNEWS_PROMPT.format(text=text)
        p_ids = tok(prompt, return_tensors="pt").input_ids[0]
        best, best_lp = -1, -1e30
        for c, lab in enumerate(label_ids):
            seq = torch.cat([p_ids, torch.tensor(lab)]).unsqueeze(0).to(device)
            logits = model(seq).logits.float()[0]
            lp = 0.0
            for k, t in enumerate(lab):
                pos = p_ids.shape[0] + k - 1
                lp += F.log_softmax(logits[pos], dim=-1)[t].item()
            lp /= len(lab)
            if lp > best_lp:
                best_lp, best = lp, c
        correct += int(best == gold)
    return correct / len(examples)


def load_model(device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    return m.to(device).eval()


def merge_all(model, calib, device, scheme, ratio):
    for li in range(len(model.model.layers)):
        block = model.model.layers[li].mlp
        K = block.top_k
        cfg = block.experts[0].config
        norm_topk = getattr(block, "norm_topk_prob", False)
        dtype = block.gate.weight.dtype
        rw = block.gate.weight.detach().float().cpu()
        freq = routing_freq(calib[li], rw, K, block.num_experts)
        plan = cluster_to_target(cosine_distance(block).cpu(), ratio)
        model.model.layers[li].mlp = MergedOlmoeMoE(
            block, plan, freq, cfg, scheme, norm_topk, device, dtype)
        del block
        gc.collect(); torch.cuda.empty_cache()
    return model


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 test + AG-News test (cached)...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_ids = tok("\n\n".join(t for t in test["text"] if t.strip()),
                   return_tensors="pt").input_ids[0]
    calib_ids = tok("\n\n".join(t for t in train["text"] if t.strip()),
                    return_tensors="pt").input_ids[0]
    ag = load_dataset("ag_news", split="test")
    N_AG = 500
    ex = [(ag[i]["text"], ag[i]["label"]) for i in range(N_AG)]
    print(f"  AG-News eval examples: {len(ex)} (chance = 25%)")

    print(f"\nLoading {MODEL_NAME}...")
    model = load_model(device)
    n_layers = len(model.model.layers)
    print("Collecting 1024-token/layer calibration...")
    calib = collect_all_layers(model, calib_ids, device, n_layers, 1024)
    print(f"  calib tokens/layer: {calib[0].shape[0]}")

    ppl0 = compute_ppl(model, test_ids, device)
    acc0 = agnews_acc(model, tok, ex, device)
    print(f"\n  baseline: PPL {ppl0:.3f}   AG-News acc {acc0*100:.1f}%")
    del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  all-layers cosine + max_logit ratio probe\n{'='*60}")
    print(f"  {'ratio':>8s} {'PPL':>10s} {'Δppl':>8s} {'AG-acc':>8s}")
    print(f"  {'base':>8s} {ppl0:>10.3f} {'--':>8s} {acc0*100:>7.1f}%")
    for ratio in (56, 48, 40):
        model = load_model(device)
        model = merge_all(model, calib, device, "max_logit", ratio)
        ppl = compute_ppl(model, test_ids, device)
        acc = agnews_acc(model, tok, ex, device)
        dppl = (ppl - ppl0) / ppl0 * 100
        print(f"  {f'64->{ratio}':>8s} {ppl:>10.3f} {dppl:>+7.1f}% "
              f"{acc*100:>7.1f}%")
        del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n  Pick the mildest ratio with usable PPL/acc for the full")
    print("  method x operator zero-shot comparison (exp 35).")


if __name__ == "__main__":
    main()
