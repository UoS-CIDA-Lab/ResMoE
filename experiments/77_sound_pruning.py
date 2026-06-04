"""Experiment 77 — Sound expert pruning via the router certificate.

Pruning expert e (removing it; in OLMoE norm_topk_prob=False so its gated term just
drops) changes a layer's output by EXACTLY g_e(h)*E_e(h) wherever e is selected. So:
  - LOSSLESS sound pruning: e is removable with ZERO change iff e is certified NEVER
    in top-K over the operating region (an L2 eps-ball union around calib). Pure router
    property: e certified-out at token t iff >=K other experts have a logit lower-bound
    (l_j - eps||w_j||) above e's upper-bound (l_e + eps||w_e||).
  - delta-BOUNDED sound pruning (the useful version): e is removable with layer-output
    impact <= delta iff max over routed inputs of ||g_e(h) E_e(h)|| <= delta. Exactly
    measurable; sound on the data (eps=0), boundable over the ball.
We report (i) lossless budget vs eps, (ii) delta-bounded budget vs delta, and
(iii) SHIFT GENERALISATION: does the delta-bounded set (chosen on prose) stay <= delta
on code+math? -- and the contrast with FREQUENCY pruning (drop least-used on calib),
which can drop an expert that shift actually relies on (a router certificate refuses to).

GPU, live OLMoE, layers {0,7,15}. Run: python3 experiments/77_sound_pruning.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
N_CALIB = 256
N_SHIFT = 256
LAYERS = [0, 7, 15]
EPS_LIST = [0.0, 0.02, 0.05, 0.10]
DELTAS = [0.005, 0.01, 0.02, 0.05, 0.1]


def swiglu(h, W1, W2, W3):
    return (F.silu(h @ W1.T) * (h @ W2.T)) @ W3.T


@torch.no_grad()
def collect(model, ids, dev, layers, n):
    cap = {li: [] for li in layers}
    hs = []
    for li in layers:
        def mk(li):
            def hk(_m, a):
                cap[li].append(a[0].detach().reshape(-1, a[0].shape[-1]))
            return hk
        hs.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    model(ids[:n].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    return {li: torch.cat(cap[li]).float() for li in layers}


def streams(tok):
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    he = load_dataset("openai_humaneval", split="test")
    gs = load_dataset("gsm8k", "main", split="test")
    return {
        "prose": tok("\n\n".join(t for t in wt["text"] if t.strip()),
                     return_tensors="pt").input_ids[0],
        "code": tok("\n\n".join(he["prompt"]), return_tensors="pt").input_ids[0],
        "math": tok("\n\n".join(gs["question"][:400]), return_tensors="pt").input_ids[0],
    }


@torch.no_grad()
def route(mlp, H):
    """gates [T,N] dense, topk [T,K], inset [T,N] over hidden states H [T,d]."""
    Wg = mlp.gate.weight.float()
    lg = H @ Wg.T
    K, N = mlp.top_k, mlp.num_experts
    topv, topi = lg.topk(K, dim=-1)
    g = torch.softmax(topv, dim=-1)
    inset = torch.zeros(H.shape[0], N, dtype=torch.bool, device=H.device)
    inset.scatter_(1, topi, True)
    gates = torch.zeros(H.shape[0], N, device=H.device)
    gates.scatter_(1, topi, g)
    return gates, inset, lg, Wg


@torch.no_grad()
def expert_contrib(mlp, e, H, idx):
    """||g_e(h) E_e(h)||_2 is computed by caller; here return ||E_e(h)||_2 over idx."""
    W1 = mlp.experts[e].gate_proj.weight.float()
    W2 = mlp.experts[e].up_proj.weight.float()
    W3 = mlp.experts[e].down_proj.weight.float()
    return swiglu(H[idx], W1, W2, W3).norm(dim=-1)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    sm = streams(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    Hp = collect(model, sm["prose"], dev, LAYERS, N_CALIB)
    Hc = collect(model, sm["code"], dev, LAYERS, N_SHIFT)
    Hm = collect(model, sm["math"], dev, LAYERS, N_SHIFT)

    for li in LAYERS:
        mlp = model.model.layers[li].mlp
        N, K = mlp.num_experts, mlp.top_k
        H = Hp[li]
        Hs = torch.cat([Hc[li], Hm[li]])
        gates, inset, lg, Wg = route(mlp, H)
        rown = Wg.norm(dim=1)
        active = [e for e in range(N) if inset[:, e].any()]
        print(f"\n{'#'*78}\n  LAYER {li}: {len(active)}/{N} experts used on prose calib "
              f"({H.shape[0]} tok)\n{'#'*78}", flush=True)

        # (i) certified-never-selected (lossless) vs eps
        print("  (i) LOSSLESS sound pruning -- certified never in top-K over eps-ball:")
        for eps in EPS_LIST:
            ub = lg + eps * rown          # [T,N] upper logit bound
            lb = lg - eps * rown          # lower
            # e certified-out at t iff >=K experts have lb_j > ub_e
            prunable = 0
            for e in range(N):
                out_all = True
                for t in range(H.shape[0]):
                    cnt = int((lb[t] > ub[t, e]).sum().item())  # experts surely above e
                    if cnt < K:           # could still be in top-K -> not certified out
                        out_all = False
                        break
                if out_all:
                    prunable += 1
            print(f"      eps={eps:<4}: {prunable:>2d} experts certified-never-selected "
                  f"({prunable/N*100:.0f}% of {N})", flush=True)

        # (ii) delta-bounded sound pruning: max_t ||g_e E_e|| over prose-routed <= delta
        contrib_max = {}                  # e -> max ||g_e E_e|| on prose (routed tokens)
        contrib_max_shift = {}
        gates_s, inset_s, _, _ = route(mlp, Hs)
        for e in active:
            idx = torch.nonzero(inset[:, e]).flatten()
            ge = gates[idx, e]
            ce = ge * expert_contrib(mlp, e, H, idx)
            contrib_max[e] = ce.max().item() if len(idx) else 0.0
            idxs = torch.nonzero(inset_s[:, e]).flatten()
            if len(idxs):
                ces = gates_s[idxs, e] * expert_contrib(mlp, e, Hs, idxs)
                contrib_max_shift[e] = ces.max().item()
            else:
                contrib_max_shift[e] = 0.0
        print("  (ii) delta-BOUNDED sound pruning -- experts with max||g_e E_e|| <= delta:")
        print(f"       {'delta':>6s} | {'prunable (prose)':>16s} | {'still<=delta on SHIFT':>22s} "
              f"| {'freq-prune same-count viol.':>26s}")
        freq = {e: int(inset[:, e].sum().item()) for e in range(N)}
        for d in DELTAS:
            sound_set = [e for e in active if contrib_max[e] <= d]
            ks = len(sound_set)
            # of the sound set, how many EXCEED delta on shift (generalisation failure)?
            shift_ok = sum(contrib_max_shift[e] <= d for e in sound_set)
            # frequency pruning: drop the ks least-frequent experts; how many of THOSE
            # exceed delta on shift (i.e. freq dropped an expert shift relies on)?
            freq_set = sorted(range(N), key=lambda e: freq[e])[:ks]
            freq_viol = sum(contrib_max_shift.get(e, 0.0) > d for e in freq_set)
            print(f"       {d:>6.3f} | {ks:>16d} | "
                  f"{shift_ok:>13d}/{ks} ok | {freq_viol:>20d}/{ks} >delta", flush=True)

    print("\n  (i) lossless budget ~0 => essentially every expert is needed somewhere")
    print("      (no free lunch), as expected. (ii) the delta-bounded budget is the real")
    print("      sound-pruning lever; SHIFT column = does the prose-certified budget hold")
    print("      out of domain; freq-prune column = how often frequency pruning drops an")
    print("      expert that EXCEEDS delta on shift (an unsafe prune the certificate refuses).")


if __name__ == "__main__":
    main()
