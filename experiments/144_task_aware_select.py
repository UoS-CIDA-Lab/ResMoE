"""Experiment 144 — TASK-aware vs FIDELITY-aware group selection (user reframe: only task
metric matters, not fidelity to original). Phi-2, all FFNs MoEfied at K=64, per-layer keep 50%,
FIXED selection (router-free) by two criteria, measured by the actual TASK metric = held-out
language-modeling loss (perplexity) + pred-divergence:
  fidelity : keep groups with largest mean ||r_g|| (FFN-output contribution)   [what we did]
  taylor   : keep groups with largest |gate * d(CE loss)/d(gate)| (effect on MODEL OUTPUT)
If taylor gives lower perplexity at the same compression => task-aware importance lets us
compress MORE at fixed task than fidelity-based selection. Reference: dense Phi-2 perplexity.
Run: python3 experiments/144_task_aware_select.py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "microsoft/phi-2"
N_CALIB = 2048
N_EVAL = 2048
CHUNK = 512
K = 64
KEEP = 0.5


def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return a


def gm_forward(mlp, x):
    sh = x.shape
    xf = x.reshape(-1, sh[-1])
    a = mlp.activation_fn(mlp.fc1(xf))
    g = mlp._gate[mlp._grp].to(a.dtype)                 # per-neuron gate from per-group gate
    out = mlp.fc2(a * g + mlp._mean * (1 - g))
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()),
              return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                 trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)

    # base eval: perplexity + argmax preds on held-out
    @torch.no_grad()
    def eval_metrics():
        ce = 0.0; ntok = 0; preds = []
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            ce += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
            preds.append(lo.argmax(-1).cpu())
        return ce / ntok, preds

    torch.set_grad_enabled(False)
    # cache per-layer fc1-input x and post-gelu a on calib
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        mlp = layers[li].mlp
        hs.append(mlp.fc1.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
        hs.append(mlp.fc2.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
    for c0 in range(0, N_CALIB, CHUNK):
        model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()

    dff = capa[0][0].shape[1]
    print(f"phi-2: {nL} layers, dff={dff}", flush=True)
    fidelity_imp = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); a = torch.cat(capa[li]).to(dev)
        mlp = layers[li].mlp
        Wup = mlp.fc1.weight.detach().float().to(dev); Wup = Wup if Wup.shape[0] == dff else Wup.T
        Wd = mlp.fc2.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
        mean_a = a.mean(0); dev_a = a - mean_a
        imp = torch.zeros(K, device=dev)
        for gi in range(K):
            ix = (grp == gi).nonzero().flatten()
            if ix.numel():
                imp[gi] = (dev_a[:, ix] @ Wd[:, ix].T).norm(dim=1).mean()
        fidelity_imp[li] = imp
        mlp._grp = grp.to(dev); mlp._mean = mean_a.half()
        mlp._gate = torch.ones(K, device=dev)
        del x, a, dev_a; gc.collect(); torch.cuda.empty_cache()
    for li in range(nL):
        layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    # TAYLOR importance: gates=1 (grad), CE loss on calib, backward, |gate*grad|
    for li in range(nL):
        layers[li].mlp._gate = torch.ones(K, device=dev, requires_grad=True)
    torch.set_grad_enabled(True)
    gopt_loss = 0.0
    for c0 in range(0, N_CALIB, CHUNK):
        xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
        lo = model(xx).logits[0, :-1].float()
        tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
        loss = F.cross_entropy(lo, tgt)
        loss.backward(); gopt_loss += loss.item()
    taylor_imp = {li: (layers[li].mlp._gate * layers[li].mlp._gate.grad).abs().detach()
                  for li in range(nL)}
    torch.set_grad_enabled(False)
    for li in range(nL):
        layers[li].mlp._gate = torch.ones(K, device=dev)

    def apply_select(imp_dict):
        kk = int(round(KEEP * K))
        for li in range(nL):
            keep_idx = imp_dict[li].topk(kk).indices
            gate = torch.zeros(K, device=dev); gate[keep_idx] = 1.0
            layers[li].mlp._gate = gate

    # dense reference (gates all 1)
    for li in range(nL):
        layers[li].mlp._gate = torch.ones(K, device=dev)
    base_ppl, base_pred = eval_metrics()
    tot = sum(p.numel() for p in base_pred)
    print(f"\n  dense Phi-2: ppl(exp CE)={torch.tensor(base_ppl).exp():.3f}  (CE={base_ppl:.4f})\n")
    print(f"  {'select':>9s} | {'CE':>7s} | {'ppl':>8s} | pred-div vs dense")
    print("  " + "-" * 48)
    for nm, imp in [("fidelity", fidelity_imp), ("taylor", taylor_imp)]:
        apply_select(imp)
        ppl, pred = eval_metrics()
        pd = sum(int((x != y).sum()) for x, y in zip(base_pred, pred)) / tot * 100
        print(f"  {nm:>9s} | {ppl:7.4f} | {torch.tensor(ppl).exp():8.3f} | {pd:6.2f}%", flush=True)
    print("\nREAD: lower ppl/CE at the SAME keep => that criterion preserves TASK better.")
    print("taylor < fidelity on ppl => task-aware importance compresses better at fixed task.")


if __name__ == "__main__":
    main()
