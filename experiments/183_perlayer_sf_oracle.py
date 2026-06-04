"""Experiment 183 — per-LAYER shared fraction (step 2): is there headroom over uniform sf?
exp176-182: shared-floor adaptive-budget MoEfy works (uniform sf=0.6 across layers). User: sf need
not be uniform per layer. Test (ORACLE diagnostic, cheap, no model re-run): per layer, for each
candidate sf, compute the global-oracle reconstruction error (neuron-level: dropped -> mean rep,
kept exact; error in down-proj output space). Pick per-layer best sf at fixed AVERAGE sf=0.6 (water-
fill: assign the sf budget where it reduces error most). Compare SUM per-layer error: uniform-sf vs
per-layer-sf. Big reduction => per-layer sf has headroom => build deployable. ~equal => inert
(convexity, like exp146-150 per-layer compression). Qwen-0.5B. Run: python3 experiments/183_...py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_CALIB = 2048
CHUNK = 512
KEEP = 0.5
SF_CANDS = [0.30, 0.45, 0.60, 0.75, 0.90]   # shared fraction of the kept budget
AVG_SF = 0.60


def recon_err(dev_a, Wd, freq, B, n_shared):
    """Global-oracle reconstruction rel-L2 in down-proj output space for a given shared split.
    shared = top-n_shared keep-freq neurons (always kept). routed budget = B - n_shared spent
    GLOBALLY across (token,neuron) by oracle contribution. dropped -> mean (dev contribution 0)."""
    N, dff = dev_a.shape
    contrib = dev_a * Wd.norm(dim=0)                       # per-(token,neuron) residual contribution
    shared_idx = freq.topk(n_shared).indices if n_shared > 0 else torch.tensor([], dtype=torch.long, device=dev_a.device)
    is_shared = torch.zeros(dff, dtype=torch.bool, device=dev_a.device); is_shared[shared_idx] = True
    rb = B - n_shared                                      # routed neurons to keep, on average
    sc = contrib.abs().clone(); sc[:, is_shared] = -1.0    # shared handled separately
    o = sc.reshape(-1).argsort(descending=True)
    keepf = torch.zeros(N * dff, dtype=torch.bool, device=dev_a.device)
    keepf[o[:N * rb]] = True                               # top N*rb routed cells globally
    sel = keepf.reshape(N, dff)
    m = sel | is_shared.unsqueeze(0)                       # kept mask (token,neuron)
    dropped = dev_a * (~m).float()                         # dropped residual contributions (dev only; mean kept)
    err = (dropped @ Wd.T).norm(dim=1)                     # ||sum_dropped dev_a_k v_k|| per token
    full = (dev_a @ Wd.T).norm(dim=1) + 1e-6
    return (err / full).mean().item()


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)

    capa = {li: [] for li in range(nL)}
    hs = [layers[li].mlp.down_proj.register_forward_pre_hook(
        (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)) for li in range(nL)]
    with torch.no_grad():
        for c0 in range(0, N_CALIB, CHUNK):
            model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]; B = int(round(KEEP * dff))
    # err[li][j] = per-layer reconstruction error at SF_CANDS[j]
    errs = torch.zeros(nL, len(SF_CANDS))
    for li in range(nL):
        a = torch.cat(capa[li]).to(dev); dev_a = a - a.mean(0)
        Wd = layers[li].mlp.down_proj.weight.detach().float().to(dev)
        Wd = Wd if Wd.shape[1] == dff else Wd.T
        contrib = dev_a.abs() * Wd.norm(dim=0)
        topB = contrib.argsort(1, descending=True)[:, :B]
        freq = torch.zeros(dff, device=dev); freq.scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev))
        for j, sf in enumerate(SF_CANDS):
            errs[li, j] = recon_err(dev_a, Wd, freq, B, int(round(sf * B)))
        del a, dev_a, Wd, contrib; gc.collect(); torch.cuda.empty_cache()
        if li % 6 == 0:
            print(f"  layer {li} errs: " + " ".join(f"{errs[li,j]:.3f}" for j in range(len(SF_CANDS))), flush=True)

    ju = SF_CANDS.index(AVG_SF)
    uniform_sum = errs[:, ju].sum().item()
    # per-layer best sf UNCONSTRAINED (lower bound on headroom)
    best_j = errs.argmin(1)
    perlayer_sum = errs[torch.arange(nL), best_j].sum().item()
    # per-layer sf CONSTRAINED to average AVG_SF (water-fill: greedily move sf where it helps, keep mean)
    sf_idx = torch.full((nL,), ju)
    cand = torch.tensor(SF_CANDS)
    for _ in range(nL * len(SF_CANDS)):
        cur_mean = cand[sf_idx].mean().item()
        best_gain = 0.0; best_li = -1; best_nj = -1
        for li in range(nL):
            for nj in range(len(SF_CANDS)):
                if nj == sf_idx[li]:
                    continue
                gain = errs[li, sf_idx[li]].item() - errs[li, nj].item()
                # keep mean ~AVG_SF: only accept moves that don't push mean away
                new_mean = cur_mean + (cand[nj] - cand[sf_idx[li]]).item() / nL
                if abs(new_mean - AVG_SF) <= abs(cur_mean - AVG_SF) + 1e-6 and gain > best_gain:
                    best_gain = gain; best_li = li; best_nj = nj
        if best_li < 0:
            break
        sf_idx[best_li] = best_nj
    constrained_sum = errs[torch.arange(nL), sf_idx].sum().item()
    eff_mean = cand[sf_idx].mean().item()

    print(f"\n  PER-LAYER SHARED FRACTION headroom (keep{KEEP}, oracle recon-err sum over {nL} layers)")
    print(f"  uniform sf={AVG_SF}:                 sum-err {uniform_sum:.3f}")
    print(f"  per-layer sf (avg {eff_mean:.2f}, constrained): sum-err {constrained_sum:.3f}  ({100*(uniform_sum-constrained_sum)/uniform_sum:+.1f}%)")
    print(f"  per-layer sf (unconstrained best):   sum-err {perlayer_sum:.3f}  ({100*(uniform_sum-perlayer_sum)/uniform_sum:+.1f}%)")
    print(f"  chosen per-layer sf: " + " ".join(f"{cand[sf_idx[li]].item():.2f}" for li in range(nL)))
    print("\nREAD: constrained per-layer sf << uniform (e.g. >5-10%) => real headroom => build deployable")
    print("per-layer sf. ~equal (<~3%) => inert (convexity, like exp146-150 per-layer compression).")


if __name__ == "__main__":
    main()
