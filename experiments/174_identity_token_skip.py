"""Experiment 174 — do IDENTITY-like tokens exist? (fair MoD test, threshold not fixed-fraction)
exp173 skipped a FIXED fraction (25/50/75%) per layer => forced over-skipping. User's hypothesis:
some tokens have FFN_out ~ 0 relative to the residual stream, so the FFN is ~identity for them and
skipping is free WITHOUT training. Fair test: per-token relative contribution ratio =
||FFN_out|| / ||residual_in||; skip only tokens with ratio < tau (VARIABLE count). Sweep tau finely.
Report (achieved skip%, ppl) + the ratio DISTRIBUTION. If a meaningful skip% stays near dense ppl
=> identity-like tokens are real & free to skip post-hoc => MoDfy viable => positioning revives.
Qwen-0.5B. Run: python3 experiments/174_identity_token_skip.py
"""
from __future__ import annotations
import sys, pathlib, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-0.5B"
N_EVAL = 4096
CHUNK = 512
TAUS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]


def mlp_forward(mlp, x):
    sh = x.shape; xf = x.reshape(-1, sh[-1])
    out = mlp.down_proj(mlp.act_fn(mlp.gate_proj(xf)) * mlp.up_proj(xf))
    rn = out.float().norm(dim=1)
    ratio = rn / (mlp._hres + 1e-6)
    mlp._ratio_buf.append(ratio.detach())
    if mlp._tau > 0:
        skip = ratio < mlp._tau
        out = out * (~skip).unsqueeze(1).to(out.dtype)
        mlp._nskip += int(skip.sum().item()); mlp._ntot += skip.numel()
    return out.reshape(sh[:-1] + (out.shape[-1],))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers; nL = len(layers)

    # hook: capture per-token residual-stream norm (input to post_attention_layernorm = FFN's residual)
    def mk_hook(li):
        def h(_m, args):
            r = args[0].reshape(-1, args[0].shape[-1]).float().norm(dim=1)
            layers[li].mlp._hres = r
        return h
    for li in range(nL):
        layers[li].post_attention_layernorm.register_forward_pre_hook(mk_hook(li))
        layers[li].mlp.forward = types.MethodType(mlp_forward, layers[li].mlp)
        layers[li].mlp._tau = 0.0

    @torch.no_grad()
    def run(tau):
        for li in range(nL):
            m = layers[li].mlp
            m._tau = tau; m._nskip = 0; m._ntot = 0; m._ratio_buf = []
        tot = 0.0; ntok = 0
        for c0 in range(0, N_EVAL, CHUNK):
            xx = ids[c0:c0 + CHUNK].unsqueeze(0).to(dev)
            lo = model(xx).logits[0, :-1].float()
            tgt = ids[c0 + 1:c0 + CHUNK].to(dev)
            tot += F.cross_entropy(lo, tgt, reduction='sum').item(); ntok += tgt.numel()
        ppl = float(torch.tensor(tot / ntok).exp())
        sk = sum(layers[li].mlp._nskip for li in range(nL))
        nt = sum(layers[li].mlp._ntot for li in range(nL))
        return ppl, (sk / nt if nt else 0.0)

    torch.set_grad_enabled(False)
    dense_ppl, _ = run(0.0)
    # ratio distribution from the tau=0 pass
    allr = torch.cat([r for li in range(nL) for r in layers[li].mlp._ratio_buf])
    qs = [0.05, 0.10, 0.20, 0.30, 0.50]
    print(f"\n  IDENTITY-like token skip (ratio = ||FFN_out|| / ||residual||). dense ppl {dense_ppl:.2f}")
    print(f"  ratio distribution over all (token,layer): " +
          ", ".join(f"p{int(q*100)}={torch.quantile(allr, q).item():.3f}" for q in qs))
    for thr in [0.005, 0.01, 0.02, 0.05, 0.10]:
        print(f"    frac with ratio<{thr}: {(allr < thr).float().mean().item()*100:.1f}%")
    print(f"\n  {'tau':>6s} | {'skip %':>7s} | {'ppl':>8s}   (vs dense {dense_ppl:.2f})")
    print("  " + "-" * 32)
    for tau in TAUS:
        ppl, sk = run(tau)
        print(f"  {tau:>6.3f} | {sk*100:>6.1f}% | {ppl:>8.2f}", flush=True)
    print("\nREAD: a skip% with ppl ~ dense => identity-like tokens are real, free to skip post-hoc")
    print("(MoD viable WITHOUT training, positioning revives). ppl rises immediately even at tiny")
    print("skip% => no free-identity tokens; even small-||out|| FFN updates are directionally needed.")


if __name__ == "__main__":
    main()
