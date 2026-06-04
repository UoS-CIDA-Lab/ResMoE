"""Experiment 38 — Faithful HC-SMoE reproduction on OLMoE: is our gap vs the
paper a PIPELINE artifact or the top-8/64 regime?

Exp 37 (hard tasks) showed OLMoE merging loses ~13% (25%) / ~30% (50%) avg
accuracy — worse than HC-SMoE's reported ~3% / ~7-13% on Qwen/Mixtral. Two
candidate causes: (a) our pipeline wasn't faithful (calibration was 2048
tok/layer + hcsmoe distance capped at 2048, vs HC-SMoE's ~65k C4 tokens), or
(b) OLMoE's top-8-of-64 routing is intrinsically harder than Qwen (top-4/60)
and Mixtral (top-2/8).

This makes the pipeline faithful: ~65k calibration tokens, full mean-expert-
output similarity (no cap), AVERAGE-linkage HC, frequency-weighted merging, and
the router-preserving ALIAS scheme — HC-SMoE's exact recipe. We accumulate
per-expert mean outputs and routing freq ONLINE over the calibration stream
(no 65k-token storage), cache the per-layer distance matrices once, then merge
+ evaluate at 25%/50% on the 4 hard tasks. Compared head-to-head with the
small-calib numbers from exp 37 (hcsmoe+alias 49.1 @25%, 40.4 @50%).

If faithful calib closes most of the gap -> pipeline; if not -> top-8 regime.
GPU capped at 75% (user constraint). Run: python3 experiments/38_...py
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
N_CALIB_TOKENS = 65536          # 32 x 2048, matching HC-SMoE
CTX = 2048


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
        gw.scatter_add_(1, groups, topv); gw = gw.to(x.dtype)
        final = torch.zeros((T, h), dtype=x.dtype, device=x.device)
        for g in range(self.r):
            sel = gw[:, g] > 0
            if sel.any():
                final[sel] += gw[sel, g, None] * self.experts[g](x[sel])
        return final.view(b, s, h), logits


@torch.no_grad()
def calibrate_online(model, ids, device, n_layers, K, N, d, n_tokens):
    """Stream calibration; accumulate per-expert mean output (HC-SMoE metric)
    and routing freq per layer WITHOUT storing all hidden states."""
    o_sum = [torch.zeros(N, d) for _ in range(n_layers)]
    freq = [torch.zeros(N) for _ in range(n_layers)]
    counts = [0] * n_layers
    seen = 0
    pos = 0
    while seen < n_tokens and pos + 1 < ids.shape[0]:
        win = ids[pos:pos + CTX]; pos += CTX
        caps = {}
        handles = []
        for li in range(n_layers):
            def mk(li):
                def hook(_m, args):
                    caps[li] = args[0].detach().reshape(-1, args[0].shape[-1])
                return hook
            handles.append(
                model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
        try:
            model(win.unsqueeze(0).to(device))
        finally:
            for hd in handles:
                hd.remove()
        for li in range(n_layers):
            H = caps[li]                                  # [T, d] on device, fp16
            block = model.model.layers[li].mlp
            logits = H @ block.gate.weight.T
            _, idx = logits.topk(K, dim=-1)
            for k in range(K):
                freq[li].scatter_add_(0, idx[:, k].cpu(),
                                      torch.ones(H.shape[0]))
            for j in range(N):
                o_sum[li][j] += block.experts[j](H).float().sum(0).cpu()
            counts[li] += H.shape[0]
        seen += win.shape[0]
        print(f"  calib window @ {seen} tokens", flush=True)
    dists = []
    for li in range(n_layers):
        o = o_sum[li] / max(counts[li], 1)                # [N, d] mean outputs
        d_li = torch.cdist(o, o)
        d_li.fill_diagonal_(0.0)
        dists.append(d_li)
    return dists, freq, counts[0]


def merge_all(model, dists, freqs, device, rmode, ratio):
    for li in range(len(model.model.layers)):
        block = model.model.layers[li].mlp
        cfg = block.experts[0].config
        dtype = block.gate.weight.dtype
        plan = _agglomerate(dists[li], ratio, "average")
        experts = build_merged_experts(block, plan, freqs[li], cfg, device, dtype)
        merged = (ReduceRouterMoE if rmode == "reduce" else AliasRouterMoE)(
            block, plan, experts, device)
        model.model.layers[li].mlp = merged
        del block
        gc.collect(); torch.cuda.empty_cache()
    return model


# ------------------------- hard zero-shot tasks --------------------------- #
@torch.no_grad()
def score_choices(model, tok, items, device, batch=16):
    seqs, meta = [], []
    for ctx, cont in items:
        cids = tok(ctx, add_special_tokens=True).input_ids
        kids = tok(cont, add_special_tokens=False).input_ids or [tok.eos_token_id]
        seqs.append((cids, kids)); meta.append((len(cids), len(kids)))
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
            lp[s + i] = sum(logp[i, clen + j - 1, full[i][clen + j]].item()
                            for j in range(klen)) / klen
    return lp


def eval_mc(model, tok, examples, device):
    flat, spans = [], []
    for ctx, conts, gold in examples:
        spans.append((len(flat), len(conts), gold))
        flat += [(ctx, c) for c in conts]
    lp = score_choices(model, tok, flat, device)
    return sum(int(max(range(nc), key=lambda j: lp[st + j]) == g)
               for st, nc, g in spans) / len(spans)


def eval_wino(model, tok, examples, device):
    flat, spans = [], []
    for (c0, p0, c1, p1), gold in examples:
        spans.append((len(flat), gold)); flat += [(c0, p0), (c1, p1)]
    lp = score_choices(model, tok, flat, device)
    return sum(int((0 if lp[st] >= lp[st + 1] else 1) == g)
               for st, g in spans) / len(spans)


def build_tasks(n=500):
    from datasets import load_dataset
    t = {}
    hs = load_dataset("hellaswag", split="validation")
    t["hellaswag"] = [((hs[i]["ctx_a"] + " " + hs[i]["ctx_b"]).strip()
                       if hs[i]["ctx_b"] else hs[i]["ctx"],
                       [" " + e for e in hs[i]["endings"]], int(hs[i]["label"]))
                      for i in range(min(n, len(hs)))]
    arc = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    t["arc_c"] = [(f"Question: {arc[i]['question']}\nAnswer:",
                   [" " + x for x in arc[i]["choices"]["text"]],
                   arc[i]["choices"]["label"].index(arc[i]["answerKey"])
                   if arc[i]["answerKey"] in arc[i]["choices"]["label"] else 0)
                  for i in range(min(n, len(arc)))]
    pq = load_dataset("piqa", split="validation")
    t["piqa"] = [(f"Question: {pq[i]['goal']}\nAnswer:",
                  [" " + pq[i]["sol1"], " " + pq[i]["sol2"]], int(pq[i]["label"]))
                 for i in range(min(n, len(pq)))]
    wg = load_dataset("winogrande", "winogrande_xl", split="validation")
    ex = []
    for i in range(min(n, len(wg))):
        pre, post = wg[i]["sentence"].split("_", 1)
        ex.append(((pre + wg[i]["option1"], post, pre + wg[i]["option2"], post),
                   int(wg[i]["answer"]) - 1))
    t["winogrande"] = ex
    return t


def run_tasks(model, tok, tasks, device):
    r = {n: eval_mc(model, tok, tasks[n], device)
         for n in ("hellaswag", "arc_c", "piqa")}
    r["winogrande"] = eval_wino(model, tok, tasks["winogrande"], device)
    r["avg"] = sum(r.values()) / len(r)
    return r


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
        print(f"GPU capped at 75% = {tot*0.75:.0f} MiB")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_ids = tok("\n\n".join(t for t in train["text"] if t.strip()),
                    return_tensors="pt").input_ids[0]
    print(f"calib pool tokens: {calib_ids.shape[0]}")
    tasks = build_tasks(500)

    print(f"\nLoading {MODEL_NAME}; faithful calibration ({N_CALIB_TOKENS} tok)...")
    model = load_model(device)
    nL = len(model.model.layers)
    b0 = model.model.layers[0].mlp
    K, N, d = b0.top_k, b0.num_experts, b0.gate.weight.shape[1]
    dists, freqs, ntok = calibrate_online(model, calib_ids, device, nL, K, N, d,
                                          N_CALIB_TOKENS)
    print(f"  calibrated on {ntok} tokens/layer")
    base = run_tasks(model, tok, tasks, device)
    print(f"  baseline avg {base['avg']*100:.1f}%  "
          + " ".join(f"{k} {v*100:.1f}" for k, v in base.items() if k != 'avg'))
    del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*72}\n  Faithful HC-SMoE (65k calib, mean-output avg-linkage,"
          f" freq-merge)\n{'='*72}")
    print(f"  {'config':22s} {'hella':>6s} {'arc_c':>6s} {'piqa':>6s} "
          f"{'wino':>6s} {'AVG':>6s}")
    print(f"  {'baseline':22s} {base['hellaswag']*100:>5.1f} {base['arc_c']*100:>5.1f}"
          f" {base['piqa']*100:>5.1f} {base['winogrande']*100:>5.1f} "
          f"{base['avg']*100:>5.1f}")
    for ratio, pct in ((48, 25), (32, 50)):
        for rmode in ("alias", "reduce"):
            model = load_model(device)
            model = merge_all(model, dists, freqs, device, rmode, ratio)
            r = run_tasks(model, tok, tasks, device)
            del model; gc.collect(); torch.cuda.empty_cache()
            print(f"  hcsmoe+{rmode} 64->{ratio}({pct}%)".ljust(24)
                  + f"{r['hellaswag']*100:>5.1f} {r['arc_c']*100:>5.1f} "
                  f"{r['piqa']*100:>5.1f} {r['winogrande']*100:>5.1f} "
                  f"{r['avg']*100:>5.1f}")

    print(f"\n{'='*72}")
    print("  vs exp 37 (2048-tok calib): hcsmoe+alias 49.1@25%, 40.4@50%.")
    print("  If faithful calib lifts these toward baseline, the gap was")
    print("  PIPELINE; if ~unchanged, it's the top-8/64 regime.")


if __name__ == "__main__":
    main()
