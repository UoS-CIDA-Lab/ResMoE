"""Experiment 35 — Full-model zero-shot accuracy: baselines x router operator.

Upgrades exp 33's single-layer rel_div comparison to the way MC-SMoE / HC-SMoE
evaluate: full OLMoE-1B-7B, all 16 layers merged, downstream zero-shot accuracy
(AG-News 4-way, log-likelihood argmax), 1024-token/layer calibration. Exp 34
showed accuracy holds through 64->48 and breaks at 64->40, so we compare at
both (48 = mild, 40 = discriminating).

Honest "vs baselines" framing (user request): the router-merge OPERATOR
(max_logit) is our drop-in contribution; we apply it to EVERY baseline
clustering (cosine / hc_smoe / freq_prune) and compare against the standard
lse_weight operator that prior MoE-merge work uses. Questions:
  - does max_logit improve the BASELINES at full-model accuracy (general win)?
  - which clustering best preserves accuracy under the good operator?

(cert-aware is omitted here: its greedy all-layers cost is prohibitive; see
exp 33 for the single-layer cert-aware comparison. mc_smoe omitted: worst in
exp 33.)

Run: python3 experiments/35_olmoe_baseline_accuracy.py
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
    d = 1.0 - vecs @ vecs.T
    d.fill_diagonal_(0.0)
    return d


@torch.no_grad()
def hc_smoe_distance(block, H, device, cap=256):
    """HC-SMoE: pairwise mean L2 between expert OUTPUTS over calib samples."""
    if H.shape[0] > cap:
        H = H[torch.randperm(H.shape[0])[:cap]]
    Hd = H.to(device=device, dtype=block.gate.weight.dtype)
    outs = torch.stack([e(Hd).float() for e in block.experts])   # [N, B, d]
    diff = outs.unsqueeze(0) - outs.unsqueeze(1)                  # [N,N,B,d]
    d = diff.norm(dim=-1).mean(dim=-1).cpu()
    d.fill_diagonal_(0.0)
    return d


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


def freq_prune_plan(freq, target_n, N):
    keep = freq.argsort(descending=True)[:target_n].tolist()
    drop = [i for i in range(N) if i not in keep]
    clusters = [[i] for i in keep] + ([drop] if drop else [])
    return MergePlan(clusters=clusters, label_of=_build_label_of(clusters, N))


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
            self.lse_rows = torch.stack([
                torch.logsumexp(block.gate.weight[m].float(), dim=0)
                for m in plan.clusters], dim=0).to(device=device,
                                                   dtype=self.orig_router.dtype)
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
            return hidden @ self.lse_rows.T
        orig = hidden @ self.orig_router.T
        cols = []
        for m in self.members:
            sub = orig.index_select(1, m)
            if self.scheme == "max_logit":
                cols.append(sub.max(dim=1).values)
            elif self.scheme == "lse_logit":
                cols.append(torch.logsumexp(sub.float(), dim=1).to(orig.dtype))
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
    """Batched length-normalized log-likelihood argmax over 4 verbalizers."""
    lab_ids = [tok(l, add_special_tokens=False).input_ids for l in AGNEWS_LABELS]
    # build all (example, label) sequences
    seqs, meta = [], []   # meta: (ex_idx, plen, llen)
    for ei, (text, gold) in enumerate(examples):
        p = tok(AGNEWS_PROMPT.format(text=text)).input_ids
        for lab in lab_ids:
            seqs.append(p + lab)
            meta.append((ei, len(p), len(lab)))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    lp = [0.0] * len(seqs)
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        ml = max(len(x) for x in chunk)
        inp = torch.full((len(chunk), ml), pad_id, dtype=torch.long)
        am = torch.zeros((len(chunk), ml), dtype=torch.long)
        for i, x in enumerate(chunk):
            inp[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
        logits = model(inp.to(device), attention_mask=am.to(device)).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        for i in range(len(chunk)):
            _, plen, llen = meta[s + i]
            tot = 0.0
            for k in range(llen):
                tok_id = chunk[i][plen + k]
                tot += logp[i, plen + k - 1, tok_id].item()
            lp[s + i] = tot / llen
    # argmax over 4 labels per example
    correct = 0
    for ei, (_, gold) in enumerate(examples):
        scores = [lp[ei * 4 + c] for c in range(4)]
        if int(max(range(4), key=lambda c: scores[c])) == gold:
            correct += 1
    return correct / len(examples)


def load_model(device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    return m.to(device).eval()


def build_plan(method, block, calib_li, freq, ratio, device):
    N = block.num_experts
    if method == "cosine":
        return cluster_to_target(cosine_distance(block).cpu(), ratio)
    if method == "hc_smoe":
        return cluster_to_target(hc_smoe_distance(block, calib_li, device), ratio)
    if method == "freq_prune":
        return freq_prune_plan(freq, ratio, N)
    raise ValueError(method)


def merge_all(model, calib, device, method, scheme, ratio):
    for li in range(len(model.model.layers)):
        block = model.model.layers[li].mlp
        K = block.top_k
        cfg = block.experts[0].config
        norm_topk = getattr(block, "norm_topk_prob", False)
        dtype = block.gate.weight.dtype
        rw = block.gate.weight.detach().float().cpu()
        freq = routing_freq(calib[li], rw, K, block.num_experts)
        plan = build_plan(method, block, calib[li], freq, ratio, device)
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
    calib = collect_all_layers(model, calib_ids, device, n_layers, 1024)
    ppl0 = compute_ppl(model, test_ids, device)
    acc0 = agnews_acc(model, tok, ex, device)
    print(f"  baseline: PPL {ppl0:.2f}  AG-acc {acc0*100:.1f}%")
    del model; gc.collect(); torch.cuda.empty_cache()

    METHODS = ("cosine", "hc_smoe", "freq_prune")
    OPS = ("lse_weight", "max_logit")
    for ratio in (48, 40):
        print(f"\n{'='*66}\n  all-16-layers 64->{ratio}  (AG-acc | PPL)\n{'='*66}")
        print(f"  {'method':12s} {'lse_weight':>20s} {'max_logit':>20s}")
        print(f"  {'':12s} {'acc      PPL':>20s} {'acc      PPL':>20s}")
        print("  " + "-" * 56)
        for method in METHODS:
            cells = {}
            for op in OPS:
                model = load_model(device)
                model = merge_all(model, calib, device, method, op, ratio)
                ppl = compute_ppl(model, test_ids, device)
                acc = agnews_acc(model, tok, ex, device)
                cells[op] = (acc, ppl)
                del model; gc.collect(); torch.cuda.empty_cache()
            print(f"  {method:12s} "
                  f"{cells['lse_weight'][0]*100:6.1f}% {cells['lse_weight'][1]:8.1f} "
                  f"   {cells['max_logit'][0]*100:6.1f}% {cells['max_logit'][1]:8.1f}")
        print(f"  {'(baseline)':12s} {acc0*100:6.1f}% {ppl0:8.1f}")

    print(f"\n{'='*66}")
    print("  READ: does max_logit beat lse_weight on ACCURACY for every")
    print("  baseline clustering (general operator win)? which clustering")
    print("  best preserves accuracy at the discriminating 64->40 ratio?")


if __name__ == "__main__":
    main()
