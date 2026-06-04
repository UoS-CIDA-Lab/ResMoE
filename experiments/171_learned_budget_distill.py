"""Experiment 171 — LEARNED adaptive budget (per-token, per-layer) via DISTILL.
Instead of fixed per-token budget, learn soft per-(layer,token,group) gates jointly with FFN
weights + rep via distillation, with a budget penalty pulling the average gate to a target keep.
The distill loss decides WHERE to spend compute (which layers/tokens get more gates on); the
penalty fixes the average. Eval with HARD gate>0.5 (=> variable per-token/layer count).
Compare to uniform-fixed-budget + distill (exp170 route-all: keep50 distill ppl 25.27, keep85 18.49)
at the SAME achieved average keep. learned < uniform => adaptive allocation beats uniform AFTER
distill (new). ~equal => distill already absorbs it / allocation inert (priors hold).
Qwen-0.5B fp32. Run: python3 experiments/171_learned_budget_distill.py
"""
from __future__ import annotations
import sys, pathlib, types, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_CALIB = 8192
N_EVAL = 2048
CHUNK = 512
K = 64
FT_STEPS = 1500
TARGETS = [0.5, 0.85]
LAM = 30.0   # budget-penalty weight


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
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    a = mlp.act_fn(mlp.gate_proj(xf)) * mlp.up_proj(xf)
    z = (xf.detach() - mlp._xbar) @ mlp._P
    g = torch.sigmoid(mlp._router(z))                 # [N, K] soft gate
    if mlp._hard:
        g = (g > 0.5).float()
    mlp._gate_mean = g.mean()
    gn = g[:, mlp._grp]
    out = mlp.down_proj(a * gn + mlp._rep * (1 - gn))
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
    def ce_eval():
        tot = 0.0; ntok = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        return tot / ntok

    @torch.no_grad()
    def measure_keep():
        set_hard(True)
        ks = []
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
            ks.append(torch.stack([layers[li].mlp._gate_mean for li in range(nL)]).mean().item())
        return sum(ks) / len(ks)

    # harvest x (gate_proj pre) for xbar/P/grp
    cap = {li: [] for li in range(nL)}
    hs = [layers[li].mlp.gate_proj.register_forward_pre_hook(
        (lambda li: (lambda _m, a: cap[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)) for li in range(nL)]
    with torch.no_grad():
        for c0 in range(0, N_CALIB, CHUNK):
            model(ids[c0:c0 + CHUNK].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    dff = layers[0].mlp.gate_proj.weight.shape[0]
    for li in range(nL):
        x = torch.cat(cap[li]).to(dev)
        xbar = x.mean(0); _, _, Vt = torch.linalg.svd(x - xbar, full_matrices=False); P = Vt[:128].T
        Wup = layers[li].mlp.gate_proj.weight.detach().float()
        grp = kmeans(F.normalize(Wup, dim=1), K, seed=0)
        mlp = layers[li].mlp
        mlp._xbar = xbar; mlp._P = P; mlp._grp = grp; mlp._hard = False
        del x; gc.collect(); torch.cuda.empty_cache()
    cap.clear(); gc.collect(); torch.cuda.empty_cache()
    for li in range(nL):
        layers[li].mlp.forward = types.MethodType(gm_forward, layers[li].mlp)

    def set_hard(h):
        for li in range(nL):
            layers[li].mlp._hard = h

    import copy
    sd0 = copy.deepcopy(model.state_dict())

    def new_router():
        return torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.GELU(),
                                   torch.nn.Linear(128, K)).to(dev).float()

    def fresh(init_bias):
        model.load_state_dict(sd0, strict=False)
        ftp = []
        for li in range(nL):
            mlp = layers[li].mlp
            r = new_router()
            with torch.no_grad():
                r[-1].bias.fill_(init_bias)            # bias>0 => gates start mostly-on
            mlp._router = r
            # rep init = mean activation (approx with 0-centered: use running; here 0 is fine since a~centered? use small)
            mlp._rep = torch.zeros(dff, device=dev, requires_grad=True)
            for mod in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                mod.weight.requires_grad_(True); ftp.append(mod.weight)
            ftp += list(r.parameters()) + [mlp._rep]
        return ftp

    @torch.no_grad()
    def dense_ppl():
        tot = 0.0; nt = 0
        for c0 in range(N_CALIB, N_CALIB + N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = dense_ref(xx).logits[0, :-1].float(); tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); nt += tgt.numel()
        return float(torch.tensor(tot / nt).exp())

    dp = dense_ppl()
    print(f"\n  LEARNED ADAPTIVE BUDGET via distill. dense ppl {dp:.2f}\n")
    print(f"  {'target':>7s} | {'achieved keep':>13s} | {'learned ppl':>11s}  (vs uniform+distill@keep)")
    print("  " + "-" * 56)
    for target in TARGETS:
        ftp = fresh(init_bias=2.0)
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
            loss = kl + LAM * (gmean - target) ** 2
            loss.backward(); torch.nn.utils.clip_grad_norm_(ftp, 1.0); opt.step(); sched.step()
        torch.set_grad_enabled(False)
        keep = measure_keep()
        set_hard(True)
        ppl = float(torch.tensor(ce_eval()).exp())
        for p in ftp:
            p.grad = None; p.requires_grad_(False)
        del opt; gc.collect(); torch.cuda.empty_cache()
        print(f"  {target:>7.2f} | {keep:>13.3f} | {ppl:>11.2f}", flush=True)
    print("\n  uniform+distill refs (exp170): keep0.5 route-all 25.27 / shared 25.79 ; keep0.85 ~18.5")
    print("READ: learned ppl < uniform+distill at matching keep => adaptive budget beats uniform after")
    print("distill. ~equal/worse => distill absorbs it; learned allocation gives nothing extra.")


if __name__ == "__main__":
    main()
