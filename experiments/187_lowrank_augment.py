"""Experiment 187 — LOW-RANK augmentation: FFN ~= static-sparse neurons + rank-r low-rank dense.
User goal: near-dense quality at <50% FLOPs. Selection alone hits the cancellation floor (exp129/143);
low-rank is an ORTHOGONAL axis AND router-free (sidesteps the deploy->oracle gap entirely). FFN out
~= down(act(gate x)*up x masked to B neurons + rep) + x @ W_r, where W_r is a rank-r linear correction
fit to the residual (full FFN out - sparse out) on calib. FLOPs: sparse B = f*FFN; low-rank r adds
2*d_model*r = r/(1.5*d_ff) of FFN. At MATCHED total ~50% FLOPs, sweep r (trade neurons for rank):
ppl drops toward dense => low-rank captures what selection can't (residual has a low-rank linear part)
=> real new lever. Flat/worse => residual is high-rank (exp129 wall holds). Qwen-0.5B.
Run: python3 experiments/187_lowrank_augment.py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_CALIB = 4096
N_EVAL = 2048
CHUNK = 512
RIDGE = 1e-2


def lr_forward(mlp, x):
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    a = mlp.act_fn(mlp.gate_proj(xf)) * mlp.up_proj(xf)
    out = mlp.down_proj(a * mlp._mask + mlp._rep * (1 - mlp._mask))   # static sparse + mean rep
    if mlp._W is not None:                                            # + rank-r low-rank dense correction
        out = out + (xf - mlp._xbar) @ mlp._W
    return out.reshape(sh[:-1] + (out.shape[-1],))


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

    @torch.no_grad()
    def ce_eval():
        tot = 0.0; nt = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float(); tg = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tg, reduction='sum').item(); nt += tg.numel()
        return tot / nt

    def ppl():
        return float(torch.tensor(ce_eval()).exp())

    dense_ppl = ppl()
    # capture FFN input x and full output o_full per layer
    capx = {li: [] for li in range(nL)}; capo = {li: [] for li in range(nL)}
    h = []
    for li in range(nL):
        h.append(layers[li].mlp.register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).detach().float().cpu())))(li)))
        h.append(layers[li].mlp.register_forward_hook(
            (lambda li: (lambda _m, _i, o: capo[li].append(o.reshape(-1, o.shape[-1]).detach().float().cpu())))(li)))
    with torch.no_grad():
        for c0 in range(0, N_CALIB, CHUNK):
            model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for hh in h:
        hh.remove()
    dff = layers[0].mlp.gate_proj.weight.shape[0]
    dmod = layers[0].mlp.gate_proj.weight.shape[1]
    LST = {}
    for li in range(nL):
        x = torch.cat(capx[li]).to(dev); ofull = torch.cat(capo[li]).to(dev)
        Wup = layers[li].mlp.gate_proj.weight.detach().float()
        Wd = layers[li].mlp.down_proj.weight.detach().float(); Wd = Wd if Wd.shape[1] == dff else Wd.T
        a = layers[li].mlp.act_fn(x @ Wup.T) * (x @ layers[li].mlp.up_proj.weight.detach().float().T)
        abar = a.mean(0)
        contrib = (a - abar).abs() * Wd.norm(dim=0)
        freq = torch.zeros(dff, device=dev)
        toph = contrib.argsort(1, descending=True)
        LST[li] = dict(x=x.cpu(), ofull=ofull.cpu(), a=a.cpu(), abar=abar, freqsort=toph.cpu(),
                       xbar=x.mean(0))
        layers[li].mlp._mean = abar
        del x, ofull, a, contrib, toph; gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0:
            print(f"  prepped {li}", flush=True)
    capx.clear(); capo.clear(); gc.collect(); torch.cuda.empty_cache()
    for li in range(nL):
        layers[li].mlp.forward = types.MethodType(lr_forward, layers[li].mlp)

    def fit_lowrank(li, B, r):
        """static top-B mask + rank-r correction of residual (full - sparse). returns mask, rep, W(or None)."""
        s = LST[li]
        x = s['x'].to(dev); a = s['a'].to(dev); ofull = s['ofull'].to(dev)
        Wd = layers[li].mlp.down_proj.weight.detach().float(); Wd = Wd if Wd.shape[1] == dff else Wd.T
        # static top-B neurons by frequency of being top-contributor
        topB = s['freqsort'][:, :B].to(dev)
        freq = torch.zeros(dff, device=dev).scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev))
        keep = freq.topk(B).indices
        mask = torch.zeros(dff, device=dev); mask[keep] = 1.0
        osparse = (a * mask + s['abar'].to(dev) * (1 - mask)) @ Wd.T
        W = None
        if r > 0:
            R = ofull - osparse                       # residual to correct
            xc = x - s['xbar'].to(dev)
            G = xc.T @ xc + RIDGE * torch.eye(dmod, device=dev)
            Wfull = torch.linalg.solve(G, xc.T @ R)   # [dmod, dmod] ridge regression
            U, S, Vt = torch.linalg.svd(Wfull, full_matrices=False)
            W = (U[:, :r] * S[:r]) @ Vt[:r]           # rank-r truncation
        del x, a, ofull, osparse; gc.collect(); torch.cuda.empty_cache()
        return mask, W

    def configure(B, r):
        for li in range(nL):
            mask, W = fit_lowrank(li, B, r)
            mlp = layers[li].mlp
            mlp._mask = mask; mlp._W = W; mlp._rep = LST[li]['abar'].to(dev); mlp._xbar = LST[li]['xbar'].to(dev)

    # FLOPs: total = B/dff + r/(1.5*dff). configs matched to ~0.50 total.
    print(f"\n  LOW-RANK AUGMENT (static-sparse + rank-r), matched ~50% FLOPs. dense ppl {dense_ppl:.2f}")
    print(f"  dff={dff} dmod={dmod}\n")
    print(f"  {'rank r':>7s} | {'keep B':>7s} | {'total FLOPs':>11s} | {'ppl':>8s}")
    print("  " + "-" * 44)
    for r in [0, 128, 256, 512]:
        f_keep = 0.50 - r / (1.5 * dff)
        B = int(round(f_keep * dff))
        configure(B, r)
        total = B / dff + r / (1.5 * dff)
        print(f"  {r:>7d} | {B/dff:>7.3f} | {total:>11.3f} | {ppl():>8.2f}", flush=True)
    print("\n  refs: pure static keep0.5 = (r=0); routed keep0.5 deploy ~64, +distill ~45 (exp179/185)")
    print("READ: ppl DROPS as r rises at fixed 50% FLOPs => low-rank captures residual selection can't")
    print("=> real lever toward near-dense. flat/worse => residual high-rank (exp129), low-rank won't save it.")


if __name__ == "__main__":
    main()
