"""Experiment 175 — can DISTILL teach the per-token FFN-skip the dense net lacks? (MoD via distill)
exp173/174: post-hoc whole-FFN skip is catastrophic & no identity tokens exist (FFN always-on).
User's reframe: the net FAILS to skip because it was never trained to — can we CHEAPLY teach it?
Per-token skip gate g=sigmoid(router(PCA(x))) per layer; train via distill (KL to dense) + budget
penalty + light FFN adapt; EVAL = hard quantile-skip lowest-g (1-c) fraction (enforces exact compute,
fixes exp171 binding). Compare distilled-MoD ppl @ compute c vs post-hoc MoD (exp173: c0.75->381,
c0.5->8643) and distilled-MoEfy (exp170: c0.85~18.5, c0.5~25). If distilled-MoD recovers to near
distilled-MoEfy => the skip is TRAINABLE (positioning revives). If it stays bad => training-locked.
Qwen-0.5B fp32. Run: python3 experiments/175_distilled_mod.py
"""
from __future__ import annotations
import sys, pathlib, types, gc, copy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_CALIB = 8192
N_EVAL = 2048
CHUNK = 512
FT_STEPS = 1500
COMPUTES = [0.75, 0.5]   # fraction of tokens that RUN the FFN
LAM = 10.0


def mod_forward(mlp, x):
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    full = mlp.down_proj(mlp.act_fn(mlp.gate_proj(xf)) * mlp.up_proj(xf))
    z = (xf.detach() - mlp._xbar) @ mlp._P
    g = torch.sigmoid(mlp._router(z).squeeze(-1))     # [N] per-token keep gate
    if mlp._hard:
        nskip = max(1, int(round((1 - mlp._ckeep) * g.numel())))
        thr = g.kthvalue(nskip).values
        mask = (g > thr).float()
        mlp._kept = mask.mean().item()
        out = full * mask.unsqueeze(1)
    else:
        mlp._gate_mean = g.mean()
        out = full * g.unsqueeze(1)
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
    dense_ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(dev).eval()
    for p in dense_ref.parameters():
        p.requires_grad_(False)
    layers = model.model.layers; nL = len(layers)

    @torch.no_grad()
    def ce_eval(usemodel):
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = usemodel(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return tot / ntok

    # harvest x (mlp input) for xbar/P
    cap = {li: [] for li in range(nL)}
    hs = [layers[li].mlp.gate_proj.register_forward_pre_hook(
        (lambda li: (lambda _m, a: cap[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)) for li in range(nL)]
    with torch.no_grad():
        for c0 in range(0, N_CALIB, CHUNK):
            model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    for li in range(nL):
        x = torch.cat(cap[li]).to(dev)
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False); P = Vt[:128].T
        mlp = layers[li].mlp
        mlp._xbar = xbar; mlp._P = P; mlp._hard = False; mlp._ckeep = 1.0
        del x; gc.collect(); torch.cuda.empty_cache()
    cap.clear(); gc.collect(); torch.cuda.empty_cache()
    for li in range(nL):
        layers[li].mlp.forward = types.MethodType(mod_forward, layers[li].mlp)

    def set_hard(h, ck=1.0):
        for li in range(nL):
            layers[li].mlp._hard = h; layers[li].mlp._ckeep = ck

    sd0 = copy.deepcopy(model.state_dict())

    def new_router():
        return torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.GELU(),
                                   torch.nn.Linear(128, 1)).to(dev).float()

    def fresh():
        model.load_state_dict(sd0, strict=False)
        ftp = []
        for li in range(nL):
            mlp = layers[li].mlp
            r = new_router()
            with torch.no_grad():
                r[-1].bias.fill_(2.0)        # start mostly-on
            mlp._router = r
            for mod in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                mod.weight.requires_grad_(True); ftp.append(mod.weight)
            ftp += list(r.parameters())
        return ftp

    @torch.no_grad()
    def meankept():
        return sum(layers[li].mlp._kept for li in range(nL)) / nL

    dense_ce = ce_eval(dense_ref)
    dp = float(torch.tensor(dense_ce).exp())
    print(f"\n  DISTILLED-MoD (learn per-token FFN skip). dense ppl {dp:.2f}")
    print(f"  refs: post-hoc MoD (exp173) c0.75->381 c0.5->8643 ; distilled-MoEfy (exp170) c0.85~18.5 c0.5~25\n")
    print(f"  {'compute c':>9s} | {'real keep':>9s} | {'distilled-MoD ppl':>17s}")
    print("  " + "-" * 44)
    for c in COMPUTES:
        ftp = fresh()
        opt = torch.optim.AdamW(ftp, lr=2e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-5, total_steps=FT_STEPS, pct_start=0.1)
        gen = torch.Generator(device=dev).manual_seed(1)
        set_hard(False)
        torch.set_grad_enabled(True)
        for st in range(FT_STEPS):
            c0 = int(torch.randint(0, N_CALIB - CHUNK, (1,), generator=gen, device=dev).item())
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            opt.zero_grad()
            sl = model(xx).logits[0, :-1].float()
            with torch.no_grad():
                dl = dense_ref(xx).logits[0, :-1].float()
            kl = F.kl_div(F.log_softmax(sl, -1), F.softmax(dl, -1), reduction='batchmean')
            gmean = torch.stack([layers[li].mlp._gate_mean for li in range(nL)]).mean()
            loss = kl + LAM * (gmean - c) ** 2
            loss.backward(); torch.nn.utils.clip_grad_norm_(ftp, 1.0); opt.step(); sched.step()
        torch.set_grad_enabled(False)
        set_hard(True, ck=c)
        ppl = float(torch.tensor(ce_eval(model)).exp()); rk = meankept()
        for p in ftp:
            p.grad = None; p.requires_grad_(False)
        del opt; gc.collect(); torch.cuda.empty_cache()
        print(f"  {c:>9.2f} | {rk:>9.3f} | {ppl:>17.2f}", flush=True)
    print("\nREAD: distilled-MoD near distilled-MoEfy (~18.5/25) => skip is TRAINABLE cheaply, the dense")
    print("net just never learned it => MoDfy+MoEfy positioning revives. Still >>MoEfy (or >>100) =>")
    print("skip is training-LOCKED (needs heavy/from-scratch, not light distill); positioning stays closed.")


if __name__ == "__main__":
    main()
