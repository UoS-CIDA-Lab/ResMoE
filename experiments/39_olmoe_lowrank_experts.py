"""Experiment 39 — Routing-orthogonal compression: low-rank experts (keep the
router + top-8 untouched).

Exps 37/38 showed expert MERGING fails at top-8/64 because it changes routing
(TS-fragility), regardless of recipe — even faithful HC-SMoE. The user's pivot:
compress each expert INTERNALLY without touching routing, so top-K stability is
irrelevant. This is the "C" step of MC-SMoE / MoE-I2's intra-expert low-rank
decomposition.

Method: for every expert, truncate each projection (gate/up/down) to rank r via
SVD, where r is chosen to hit the SAME parameter-reduction level as the merge
experiments (25% / 50%). The router and top-8 selection are EXACTLY unchanged,
so the model runs with the stock OlmoeSparseMoeBlock forward. Numerically the
truncated-dense weight equals a factored U,V implementation, so accuracy here
reflects the real low-rank model (memory saving is implied by r).

Param fraction kept for a matrix [m,n] at rank r is r(m+n)/(mn). For OLMoE
gate/up [1024,2048] and down [2048,1024], 50% keep -> r=341, 75% keep -> r=512.

Compared head-to-head (same harness/tasks/calib-free) with merging at 25%/50%
from exp 37/38 (hcsmoe best: 48.9 @25%, 42.8 @50% avg; baseline 57.5).
Prediction: low-rank preserves accuracy far better because routing is intact.
GPU capped at 75%. Run: python3 experiments/39_olmoe_lowrank_experts.py
"""
from __future__ import annotations

import sys
import pathlib
import gc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


def rank_for_fraction(m, n, keep_frac):
    """Rank r so r(m+n)/(mn) ~= keep_frac; clamped to [1, min(m,n)]."""
    r = round(keep_frac * m * n / (m + n))
    return max(1, min(r, min(m, n)))


@torch.no_grad()
def truncate_lowrank(weight, r):
    """Return rank-r SVD truncation of a 2-D weight (dense, same shape)."""
    W = weight.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Wr = (U[:, :r] * S[:r]) @ Vh[:r]
    return Wr.to(weight.dtype)


@torch.no_grad()
def lowrank_all(model, keep_frac, device):
    """Truncate every expert projection to the rank matching keep_frac."""
    for layer in model.model.layers:
        block = layer.mlp
        for e in block.experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                W = getattr(e, proj).weight
                m, n = W.shape
                r = rank_for_fraction(m, n, keep_frac)
                getattr(e, proj).weight.copy_(truncate_lowrank(W, r))
        gc.collect(); torch.cuda.empty_cache()
    return model


def report_ranks(model):
    block = model.model.layers[0].mlp.experts[0]
    lines = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        m, n = getattr(block, proj).weight.shape
        lines.append(f"{proj}[{m}x{n}]")
    return ", ".join(lines)


# ------------------------- hard zero-shot tasks (exp 37/38) --------------- #
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


@torch.no_grad()
def compute_ppl(model, ids, device, ctx=1024, max_windows=30):
    import math
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
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_ids = tok("\n\n".join(t for t in test["text"] if t.strip()),
                   return_tensors="pt").input_ids[0]
    tasks = build_tasks(500)

    print(f"\nLoading {MODEL_NAME} (baseline)...")
    model = load_model(device)
    print(f"  expert projections: {report_ranks(model)}")
    base = run_tasks(model, tok, tasks, device)
    base_ppl = compute_ppl(model, test_ids, device)
    print(f"  baseline avg {base['avg']*100:.1f}%  PPL {base_ppl:.2f}")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ranks for the report
    for pct in (25, 50):
        kf = 1 - pct / 100
        r_gu = rank_for_fraction(1024, 2048, kf)
        r_dn = rank_for_fraction(2048, 1024, kf)
        print(f"  {pct}% reduction -> keep {kf:.2f}: gate/up rank {r_gu}/1024, "
              f"down rank {r_dn}/1024")

    print(f"\n{'='*74}\n  Routing-orthogonal low-rank experts (router + top-8 "
          f"UNTOUCHED)\n{'='*74}")
    print(f"  {'config':22s} {'hella':>6s} {'arc_c':>6s} {'piqa':>6s} "
          f"{'wino':>6s} {'AVG':>6s} {'PPL':>8s}")
    print(f"  {'baseline':22s} {base['hellaswag']*100:>5.1f} {base['arc_c']*100:>5.1f}"
          f" {base['piqa']*100:>5.1f} {base['winogrande']*100:>5.1f} "
          f"{base['avg']*100:>5.1f} {base_ppl:>8.2f}")
    for pct in (25, 50):
        kf = 1 - pct / 100
        model = load_model(device)
        model = lowrank_all(model, kf, device)
        r = run_tasks(model, tok, tasks, device)
        ppl = compute_ppl(model, test_ids, device)
        del model; gc.collect(); torch.cuda.empty_cache()
        print(f"  {'lowrank ' + str(pct) + '% reduction':22s} "
              f"{r['hellaswag']*100:>5.1f} {r['arc_c']*100:>5.1f} "
              f"{r['piqa']*100:>5.1f} {r['winogrande']*100:>5.1f} "
              f"{r['avg']*100:>5.1f} {ppl:>8.2f}")

    print(f"\n{'='*74}")
    print("  vs MERGING at same reduction (exp 37/38, same harness): hcsmoe")
    print("  best avg 48.9@25%, 42.8@50% (baseline 57.5). If low-rank keeps")
    print("  avg far higher, routing-orthogonal compression is the right knob")
    print("  for top-8 MoE — merging's TS-fragility is avoided entirely.")


if __name__ == "__main__":
    main()
