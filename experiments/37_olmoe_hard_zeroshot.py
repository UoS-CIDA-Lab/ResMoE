"""Experiment 37 — Discriminating zero-shot tasks (HC-SMoE's actual benchmarks).

The user correctly noted AG-News topic classification is too easy / insensitive
to certify that merged models retain real capability (PPL rose 3-1000x while
AG-acc barely moved, and freq_prune even "beat" baseline = noise). This re-
evaluates on the HARD multiple-choice tasks HC-SMoE uses, which are not solvable
by surface lexical cues and degrade sharply with model damage:
  HellaSwag (commonsense continuation), ARC-Challenge (science reasoning),
  WinoGrande (coreference), PIQA (physical commonsense).
Metric = length-normalized log-likelihood argmax (lm-eval acc protocol), and
their average — exactly how HC-SMoE reports (Table 2/3).

Compares baseline vs merged (all-16-layers) at 64->48 (25%) and 64->32 (50%),
for the decisive configs, reusing exp 36's merge machinery (cosine/hcsmoe
clustering x reduce/alias router). GPU capped at 75% (user constraint).

Run: python3 experiments/37_olmoe_hard_zeroshot.py
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


# ----------------------------- merge machinery (from exp 36) -------------- #
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
    if H.shape[0] > cap:
        H = H[torch.randperm(H.shape[0])[:cap]]
    Hd = H.to(device=device, dtype=block.gate.weight.dtype)
    o = torch.stack([block.experts[i](Hd).float().mean(0)
                     for i in range(block.num_experts)])
    d = torch.cdist(o, o).cpu()
    d.fill_diagonal_(0.0)
    return d


def routing_freq(H, router_weight, K, n_experts):
    logits = H @ router_weight.T
    _, idx = logits.topk(K, dim=-1)
    freq = torch.zeros(n_experts)
    for k in range(K):
        freq.scatter_add_(0, idx[:, k], torch.ones(H.shape[0]))
    return freq


def _agglomerate(dist, target_n, linkage):
    n0 = dist.shape[0]
    D = dist.clone().float(); D.fill_diagonal_(math.inf)
    clusters = [[i] for i in range(n0)]; sizes = [1] * n0
    while len(clusters) > target_n:
        idx = int(D.argmin().item()); a, b = divmod(idx, D.shape[0])
        if a > b:
            a, b = b, a
        if linkage == "complete":
            nr = torch.maximum(D[a], D[b])
        else:
            nr = (sizes[a] * D[a] + sizes[b] * D[b]) / (sizes[a] + sizes[b])
        D[a] = nr; D[:, a] = nr; D[a, a] = math.inf
        keep = [i for i in range(D.shape[0]) if i != b]; D = D[keep][:, keep]
        clusters[a] = clusters[a] + clusters[b]; sizes[a] += sizes[b]
        del clusters[b]; del sizes[b]
    return MergePlan(clusters=clusters, label_of=_build_label_of(clusters, n0))


def build_merged_experts(block, plan, freq, cfg, device, dtype):
    from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
    experts = nn.ModuleList()
    with torch.no_grad():
        for members in plan.clusters:
            w = freq[members].float()
            w = w / w.sum() if w.sum() > 0 else torch.ones_like(w) / len(w)
            e = OlmoeMLP(cfg).to(device=device, dtype=dtype)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                acc = sum(w[i] * getattr(block.experts[m], proj).weight.float()
                          for i, m in enumerate(members))
                getattr(e, proj).weight.copy_(acc.to(dtype))
            experts.append(e)
    return experts


class ReduceRouterMoE(nn.Module):
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
                final.index_add_(0, tx, (self.experts[c](x[tx])
                                         * routing[tx, i, None]).to(final.dtype))
        return final.view(b, s, h), cl


class AliasRouterMoE(nn.Module):
    def __init__(self, block, plan, experts, device):
        super().__init__()
        self.norm_topk_prob = getattr(block, "norm_topk_prob", False)
        self.K = block.top_k
        self.r = len(plan.clusters)
        self.router = block.gate.weight.detach().clone().to(device)
        self.label_of = torch.tensor(plan.label_of, device=device)
        self.experts = experts

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h); T = x.shape[0]
        logits = x @ self.router.T
        probs = F.softmax(logits, dim=1, dtype=torch.float)
        topv, topi = probs.topk(self.K, dim=1)
        if self.norm_topk_prob:
            topv = topv / topv.sum(dim=1, keepdim=True)
        groups = self.label_of[topi]
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
    caps = {li: [] for li in range(n_layers)}; handles = []
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


def merge_all(model, calib, device, clustering, rmode, ratio):
    for li in range(len(model.model.layers)):
        block = model.model.layers[li].mlp
        cfg = block.experts[0].config
        dtype = block.gate.weight.dtype
        rw = block.gate.weight.detach().float().cpu()
        freq = routing_freq(calib[li], rw, block.top_k, block.num_experts)
        if clustering == "cosine":
            plan = _agglomerate(cosine_distance(block).cpu(), ratio, "complete")
        else:
            plan = _agglomerate(hcsmoe_distance(block, calib[li], device),
                                ratio, "average")
        experts = build_merged_experts(block, plan, freq, cfg, device, dtype)
        merged = (ReduceRouterMoE if rmode == "reduce" else AliasRouterMoE)(
            block, plan, experts, device)
        model.model.layers[li].mlp = merged
        del block
        gc.collect(); torch.cuda.empty_cache()
    return model


# ----------------------------- zero-shot scoring -------------------------- #
@torch.no_grad()
def score_choices(model, tok, items, device, batch=16):
    """items: list of (ctx_str, cont_str, n_choices, gold). Returns accuracy.
    Length-normalized log-likelihood of cont given ctx; argmax per group."""
    seqs, meta = [], []
    for ctx, cont, _, _ in items:
        cids = tok(ctx, add_special_tokens=True).input_ids
        kids = tok(cont, add_special_tokens=False).input_ids
        if len(kids) == 0:
            kids = [tok.eos_token_id]
        seqs.append((cids, kids))
        meta.append((len(cids), len(kids)))
    lp = [0.0] * len(seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        full = [c + k for c, k in chunk]
        ml = max(len(x) for x in full)
        inp = torch.full((len(full), ml), pad, dtype=torch.long)
        am = torch.zeros((len(full), ml), dtype=torch.long)
        for i, x in enumerate(full):
            inp[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
        logp = F.log_softmax(
            model(inp.to(device), attention_mask=am.to(device)).logits.float(),
            dim=-1)
        for i in range(len(chunk)):
            clen, klen = meta[s + i]
            tot = 0.0
            for j in range(klen):
                tot += logp[i, clen + j - 1, full[i][clen + j]].item()
            lp[s + i] = tot / klen
    return lp  # per-item length-normalized loglik; caller assembles by group


def eval_task(model, tok, examples, device):
    """examples: list of (ctx, [cont0,cont1,...], gold). Returns accuracy."""
    flat, spans = [], []
    for ctx, conts, gold in examples:
        spans.append((len(flat), len(conts), gold))
        for c in conts:
            flat.append((ctx, c, 0, 0))
    lp = score_choices(model, tok, flat, device)
    correct = 0
    for start, nc, gold in spans:
        scores = lp[start:start + nc]
        if max(range(nc), key=lambda j: scores[j]) == gold:
            correct += 1
    return correct / len(spans)


def build_tasks(tok, n=500):
    from datasets import load_dataset
    tasks = {}

    hs = load_dataset("hellaswag", split="validation")
    ex = []
    for i in range(min(n, len(hs))):
        r = hs[i]
        ctx = (r["ctx_a"] + " " + r["ctx_b"]).strip() if r["ctx_b"] else r["ctx"]
        ex.append((ctx, [" " + e for e in r["endings"]], int(r["label"])))
    tasks["hellaswag"] = ex

    arc = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    ex = []
    for i in range(min(n, len(arc))):
        r = arc[i]
        labels = r["choices"]["label"]
        gold = labels.index(r["answerKey"]) if r["answerKey"] in labels else 0
        ctx = f"Question: {r['question']}\nAnswer:"
        ex.append((ctx, [" " + t for t in r["choices"]["text"]], gold))
    tasks["arc_c"] = ex

    pq = load_dataset("piqa", split="validation")
    ex = []
    for i in range(min(n, len(pq))):
        r = pq[i]
        ctx = f"Question: {r['goal']}\nAnswer:"
        ex.append((ctx, [" " + r["sol1"], " " + r["sol2"]], int(r["label"])))
    tasks["piqa"] = ex

    wg = load_dataset("winogrande", "winogrande_xl", split="validation")
    ex = []
    for i in range(min(n, len(wg))):
        r = wg[i]
        # score the suffix after the blank, conditioned on prefix+option
        pre, post = r["sentence"].split("_", 1)
        gold = int(r["answer"]) - 1
        ex.append(((pre + r["option1"], post, pre + r["option2"], post), None, gold))
    tasks["winogrande"] = ex
    return tasks


def eval_winogrande(model, tok, examples, device):
    flat, spans = [], []
    for (c0, p0, c1, p1), _, gold in examples:
        spans.append((len(flat), gold))
        flat.append((c0, p0, 0, 0)); flat.append((c1, p1, 0, 0))
    lp = score_choices(model, tok, flat, device)
    correct = 0
    for start, gold in spans:
        sc = lp[start:start + 2]
        correct += int((0 if sc[0] >= sc[1] else 1) == gold)
    return correct / len(spans)


def run_all_tasks(model, tok, tasks, device):
    res = {}
    for name in ("hellaswag", "arc_c", "piqa"):
        res[name] = eval_task(model, tok, tasks[name], device)
    res["winogrande"] = eval_winogrande(model, tok, tasks["winogrande"], device)
    res["avg"] = sum(res.values()) / len(res)
    return res


def load_model(device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    return m.to(device).eval()


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"GPU capped at 75% of {tot:.0f} MiB = {tot*0.75:.0f} MiB")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_ids = tok("\n\n".join(t for t in train["text"] if t.strip()),
                    return_tensors="pt").input_ids[0]
    print("Building hard zero-shot tasks (500 ex each)...")
    tasks = build_tasks(tok, n=500)
    for k, v in tasks.items():
        print(f"  {k}: {len(v)} examples")

    print(f"\nLoading {MODEL_NAME}...")
    model = load_model(device)
    n_layers = len(model.model.layers)
    calib = collect_all_layers(model, calib_ids, device, n_layers, 2048)
    base = run_all_tasks(model, tok, tasks, device)
    print(f"\n  baseline: " + "  ".join(f"{k} {v*100:.1f}%" for k, v in base.items()))
    del model; gc.collect(); torch.cuda.empty_cache()

    # decisive configs (clustering, router)
    configs = [("cosine", "reduce"), ("hcsmoe", "reduce"), ("hcsmoe", "alias")]
    for ratio, pct in ((48, 25), (32, 50)):
        print(f"\n{'='*78}\n  all-16-layers 64->{ratio} ({pct}%)\n{'='*78}")
        print(f"  {'config':18s} {'hella':>7s} {'arc_c':>7s} {'piqa':>7s} "
              f"{'wino':>7s} {'AVG':>7s}")
        print("  " + "-" * 60)
        print(f"  {'baseline':18s} {base['hellaswag']*100:>6.1f}% "
              f"{base['arc_c']*100:>6.1f}% {base['piqa']*100:>6.1f}% "
              f"{base['winogrande']*100:>6.1f}% {base['avg']*100:>6.1f}%")
        for clustering, rmode in configs:
            model = load_model(device)
            model = merge_all(model, calib, device, clustering, rmode, ratio)
            r = run_all_tasks(model, tok, tasks, device)
            del model; gc.collect(); torch.cuda.empty_cache()
            print(f"  {clustering+'+'+rmode:18s} {r['hellaswag']*100:>6.1f}% "
                  f"{r['arc_c']*100:>6.1f}% {r['piqa']*100:>6.1f}% "
                  f"{r['winogrande']*100:>6.1f}% {r['avg']*100:>6.1f}%")

    print(f"\n{'='*78}")
    print("  HellaSwag/ARC-c chance=25%, PIQA/Wino=50%. These are sensitive to")
    print("  real capability loss (unlike AG-News). Does ANY merge config keep")
    print("  the AVG near baseline at 50%, as HC-SMoE claims for Qwen/Mixtral?")


if __name__ == "__main__":
    main()
