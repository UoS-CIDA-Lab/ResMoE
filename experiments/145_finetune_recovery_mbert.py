"""Experiment 145 — FINE-TUNING RECOVERY (the real task-preservation lever; user goal: only
task matters, compress more). mBERT (G-MoE baseline, cheap to finetune). MoEfy ALL FFNs at
aggressive keep (fixed per-layer selection by fidelity ||r_g||, representative=mean), then
FINE-TUNE the retained FFN weights + representative on the MLM loss. Measure the TASK metric
(MLM cross-entropy on held-out) for dense / static(no-ft) / fine-tuned, at keep 50% & 25%.
Q: does fine-tuning recover task enough that aggressive removal is ~task-lossless (G-MoE's
secret sauce), i.e. compress more at fixed task than static MoEfy?
Run: python3 experiments/145_finetune_recovery_mbert.py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

NAME = "bert-base-multilingual-cased"
N_CALIB = 16384      # tokens for stats + finetune
N_EVAL = 8192
SEQ = 512
K = 64
KEEPS = [0.5, 0.25]
FT_STEPS = 400
MASK_P = 0.15


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


def patched_intermediate(self, hidden_states):
    h = self.dense(hidden_states)
    a = self.intermediate_act_fn(h)
    m = self._mask
    return a * m + self._rep * (1 - m)


def main():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids_all = tok("\n\n".join(t for t in wt["text"] if t.strip()),
                  return_tensors="pt").input_ids[0]
    mask_id = tok.mask_token_id
    model = AutoModelForMaskedLM.from_pretrained(NAME).to(dev).eval()
    bert = model.bert
    layers = bert.encoder.layer
    nL = len(layers)

    def batches(start, n):
        out = []
        for c0 in range(start, start + n, SEQ):
            out.append(ids_all[c0:c0 + SEQ].unsqueeze(0).to(dev))
        return out

    @torch.no_grad()
    def mlm_ce(seed=0):
        gen = torch.Generator(device=dev).manual_seed(seed)
        tot = 0.0; ntok = 0
        for xb in batches(N_CALIB, N_EVAL):
            inp = xb.clone()
            pm = (torch.rand(inp.shape, generator=gen, device=dev) < MASK_P)
            pm[:, 0] = False
            lab = inp.clone(); lab[~pm] = -100
            inp[pm] = mask_id
            lo = model(inp).logits[0]
            l = F.cross_entropy(lo.float(), lab[0], ignore_index=-100, reduction='sum')
            tot += l.item(); ntok += (lab[0] != -100).sum().item()
        return tot / max(ntok, 1)

    # cache per-layer intermediate input(x) and post-gelu(a) on calib
    torch.set_grad_enabled(False)
    capx = {li: [] for li in range(nL)}; capa = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        inter = layers[li].intermediate
        hs.append(inter.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
        hs.append(inter.register_forward_hook(
            (lambda li: (lambda _m, _i, o: capa[li].append(o.reshape(-1, o.shape[-1]).float())))(li)))
    for xb in batches(0, N_CALIB):
        model(xb)
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    print(f"mBERT: {nL} layers, dff={dff}", flush=True)

    # per-layer groups + fidelity importance; attach mask/rep
    for li in range(nL):
        inter = layers[li].intermediate
        Wup = inter.dense.weight.detach().float()                  # [dff, d]
        Wd = layers[li].output.dense.weight.detach().float()       # [d, dff]
        a = torch.cat(capa[li]).to(dev); mean_a = a.mean(0); dev_a = a - mean_a
        grp = kmeans(F.normalize(Wup, dim=1).to(dev), K, seed=0)
        imp = torch.zeros(K, device=dev)
        for gi in range(K):
            ix = (grp == gi).nonzero().flatten()
            if ix.numel():
                imp[gi] = (dev_a[:, ix] @ Wd[:, ix].T.to(dev)).norm(dim=1).mean()
        inter._grp = grp; inter._imp = imp; inter._mean = mean_a.to(model.dtype)
        inter._rep0 = mean_a.to(model.dtype).clone()
        inter.intermediate_act_fn = inter.intermediate_act_fn  # keep
        inter.forward = types.MethodType(patched_intermediate, inter)
        del a, dev_a; gc.collect(); torch.cuda.empty_cache()

    def set_keep(keep, reset_rep=True):
        kk = int(round(keep * K))
        for li in range(nL):
            inter = layers[li].intermediate
            keep_idx = inter._imp.topk(kk).indices
            gate = torch.zeros(K, device=dev); gate[keep_idx] = 1.0
            inter._mask = gate[inter._grp].to(model.dtype)         # [dff] 0/1
            if reset_rep:
                inter._rep = inter._rep0.clone()

    def set_dense():
        for li in range(nL):
            inter = layers[li].intermediate
            inter._mask = torch.ones(dff, device=dev, dtype=model.dtype)
            inter._rep = inter._rep0
    set_dense()
    print(f"\n  dense mBERT MLM-CE = {mlm_ce():.4f}\n")
    print(f"  {'keep':>5s} | {'static-CE':>9s} | {'finetuned-CE':>12s} | (dense ref)")
    print("  " + "-" * 46)
    import copy
    sd0 = copy.deepcopy(model.state_dict())
    for keep in KEEPS:
        # reload clean weights each keep
        model.load_state_dict(sd0)
        for li in range(nL):  # rebind patched forward after load
            inter = layers[li].intermediate
            inter.forward = types.MethodType(patched_intermediate, inter)
        set_keep(keep)
        ce_static = mlm_ce()
        # fine-tune: retained FFN weights + representative
        for p in model.parameters():
            p.requires_grad_(False)
        ft_params = []
        for li in range(nL):
            inter = layers[li].intermediate
            inter.dense.weight.requires_grad_(True)
            layers[li].output.dense.weight.requires_grad_(True)
            inter._rep = inter._rep0.clone().requires_grad_(True)
            ft_params += [inter.dense.weight, layers[li].output.dense.weight, inter._rep]
        opt = torch.optim.AdamW(ft_params, lr=2e-5)
        gen = torch.Generator(device=dev).manual_seed(1)
        tb = batches(0, N_CALIB)
        torch.set_grad_enabled(True)
        for st in range(FT_STEPS):
            xb = tb[st % len(tb)]
            inp = xb.clone()
            pm = (torch.rand(inp.shape, generator=gen, device=dev) < MASK_P); pm[:, 0] = False
            lab = inp.clone(); lab[~pm] = -100; inp[pm] = mask_id
            opt.zero_grad()
            lo = model(inp).logits[0]
            loss = F.cross_entropy(lo.float(), lab[0], ignore_index=-100)
            loss.backward(); torch.nn.utils.clip_grad_norm_(ft_params, 1.0); opt.step()
        torch.set_grad_enabled(False)
        ce_ft = mlm_ce()
        print(f"  {int(keep*100):>4d}% | {ce_static:9.4f} | {ce_ft:12.4f} |", flush=True)
    print("\nREAD: finetuned-CE close to dense => fine-tuning recovers task after aggressive removal")
    print("(G-MoE's mechanism). Big static->ft gap => the task headroom is in FINE-TUNING, not selection.")


if __name__ == "__main__":
    main()
