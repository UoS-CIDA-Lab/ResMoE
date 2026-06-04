"""Experiment 127 — Domain-specific MoEfication: is a gated FFN much more compressible on a
NARROW domain than on the general distribution?

exp 107-126 all calib==eval on WikiText (general). The ~38% @ eff .5 frontier is a GENERAL-
domain number. exp 125 tried to PARTITION the general distribution into input regions and
failed (curse of dimensionality + data starvation). Domain-MoEfy is different: commit to ONE
coherent real domain (no partition, plenty of data). Hypothesis: a dense FFN trained for ALL
domains is over-provisioned for any single one -> on a narrow domain the activation lives in
a lower-dim subspace, so a FIXED domain prune-set (drop what the domain doesn't use) is far
better than general pruning, maybe approaching the per-token oracle (-> domain MoEfy needs no
routing). This is the natural deployment story (deploy-for-code -> MoEfy-for-code) and the one
removal angle not yet measured at the FFN level.

Domains: prose=WikiText, code=MBPP, math=GSM8K. OLMoE experts as dense SwiGLU, per-UNIT
(each neuron its own group), keep k, rep=domain mean. Per domain D (eval on D):
  effrank90    : # PCA comps of activation deviation for 90% energy (lower = more compressible)
  prune(dom)   : FIXED top-h units by D-importance, D-mean rep                 (domain MoEfy)
  prune(gen)   : FIXED top-h units by GENERAL-importance + general mean        (general MoEfy)
  oracle(dom)  : per-token top-h units, D-mean rep                            (per-unit floor)

READ: narrow domains with LOW effrank + prune(dom) << prune(gen) + prune(dom) ~ oracle(dom)
=> domain MoEfy works: statically drop what the domain doesn't use, no routing needed (first
constructive POSITIVE). prune(dom) ~ general frontier => no domain headroom; the FFN uses its
full capacity even per-domain. Run: python3 experiments/127_domain_moefy.py
"""
from __future__ import annotations

import sys
import pathlib
import statistics as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

MODEL = "allenai/OLMoE-1B-7B-0924"
LAYER = 0
N_TOK = 6000
N_EXPERTS = 4
KEEP = [0.5, 0.85]
CALIB_FRAC = 0.6


def build_domains(tok):
    from datasets import load_dataset
    out = {}
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    out["prose"] = "\n\n".join(t for t in wt["text"] if t.strip())
    mb = load_dataset("mbpp", "full", split="train")
    out["code"] = "\n\n".join(mb["code"])
    gs = load_dataset("gsm8k", "main", split="train")
    out["math"] = "\n\n".join(q + "\n" + a for q, a in zip(gs["question"], gs["answer"]))
    return {k: tok(v, return_tensors="pt").input_ids[0] for k, v in out.items()}


def collect_inputs(model, ids, dev, layer, n, win=2048):
    cap = []
    h = model.model.layers[layer].mlp.register_forward_pre_hook(
        lambda _m, a: cap.append(a[0].detach().reshape(-1, a[0].shape[-1])))
    got = 0
    with torch.no_grad():
        for s in range(0, ids.shape[0] - 1, win):
            model(ids[s:s + win].unsqueeze(0).to(dev))
            got += min(win, ids.shape[0] - s)
            if got >= n:
                break
    h.remove()
    return torch.cat(cap)[:n].float()


def effrank90(M):                       # # singular comps of M for 90% energy
    s = torch.linalg.svdvals(M)
    c = (s ** 2).cumsum(0) / (s ** 2).sum()
    return int((c < 0.90).sum().item()) + 1


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    dom_ids = build_domains(tok)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    HD = {name: collect_inputs(model, ids, dev, LAYER, N_TOK)
          for name, ids in dom_ids.items()}
    mlp = model.model.layers[LAYER].mlp
    d = list(HD.values())[0].shape[1]
    EW = {e: (mlp.experts[e].gate_proj.weight.detach().float(),
              mlp.experts[e].up_proj.weight.detach().float(),
              mlp.experts[e].down_proj.weight.detach().float())
          for e in range(N_EXPERTS)}
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    # split each domain; general calib = concat of per-domain calibs
    DC, DE = {}, {}
    for name, H in HD.items():
        p = torch.randperm(H.shape[0], device=dev)
        nc = int(CALIB_FRAC * H.shape[0])
        DC[name], DE[name] = H[p[:nc]], H[p[nc:]]
    allc = torch.cat([DC[n] for n in DC])
    gen_c = allc[torch.randperm(allc.shape[0], device=dev)][:max(len(c) for c in DC.values())]

    domains = list(HD.keys())
    print(f"domain MoEfy: layer {LAYER}, {N_EXPERTS} experts as dense SwiGLU, d={d}, "
          f"dff=1024; per-unit; tokens/domain~{N_TOK}\n")
    # acc[(metric, domain, keep)] -> list over experts*tokens
    acc = {}
    er = {nm: [] for nm in domains}

    for e in EW:
        W1e, W2e, W3e = EW[e]
        w3n = W3e.norm(dim=0)

        def acts(X):
            return F.silu(X @ W1e.T) * (X @ W2e.T)

        # general importance/mean (built on mixed calib)
        ug = acts(gen_c)
        gmean = ug.mean(0)
        gimp = (ug - gmean).abs().mean(0) * w3n           # general per-unit importance

        for nm in domains:
            Xc, Xe = DC[nm], DE[nm]
            uc, ue = acts(Xc), acts(Xe)
            dmean = uc.mean(0)
            Ye = ue @ W3e.T
            yn = Ye.norm(dim=1).clamp(min=1e-8)
            er[nm].append(effrank90(ue - dmean))
            dimp = (uc - dmean).abs().mean(0) * w3n        # domain per-unit importance
            ne, dff = ue.shape

            def err(mask, rep):
                kept = ue * mask + rep * (1 - mask)
                return ((kept @ W3e.T - Ye).norm(dim=1) / yn * 100).tolist()

            for kp in KEEP:
                hh = max(1, round(kp * dff))
                # FIXED domain prune
                kd = dimp.topk(hh).indices
                md = torch.zeros(ne, dff, device=dev); md[:, kd] = 1.0
                acc.setdefault(("prune-dom", nm, kp), []).extend(err(md, dmean))
                # FIXED general prune (general units + general mean), eval on D
                kg = gimp.topk(hh).indices
                mg = torch.zeros(ne, dff, device=dev); mg[:, kg] = 1.0
                acc.setdefault(("prune-gen", nm, kp), []).extend(err(mg, gmean))
                # per-token oracle (per-unit floor on D)
                imp_e = (ue - dmean).abs() * w3n
                ko = imp_e.topk(hh, dim=1).indices
                mo = torch.zeros(ne, dff, device=dev); mo.scatter_(1, ko, 1.0)
                acc.setdefault(("oracle-dom", nm, kp), []).extend(err(mo, dmean))
        print(f"  expert {e} done", flush=True)

    def med(metric, nm, kp):
        return st.median(acc[(metric, nm, kp)])

    print(f"  {'domain':>7s} {'effrank90':>9s} |" +
          "".join(f"  keep{int(k*100)}%: prune-dom / prune-gen / oracle-dom" for k in KEEP))
    print("  " + "-" * 86)
    for nm in domains:
        cells = []
        for kp in KEEP:
            cells.append(f"{med('prune-dom', nm, kp):>5.1f} / "
                         f"{med('prune-gen', nm, kp):>5.1f} / {med('oracle-dom', nm, kp):>5.1f}")
        print(f"  {nm:>7s} {st.mean(er[nm]):>9.1f} |   " + "      ".join(cells))
    print("\n  (general WikiText frontier, same metric/per-unit-ish: G-MoE ~38% @ eff .5)")
    print("\nREAD: low effrank + prune-dom << prune-gen + prune-dom ~ oracle-dom => domain")
    print("MoEfy works (static domain drop, no routing). prune-dom ~ general/prune-gen =>")
    print("no domain headroom; the FFN uses full capacity even per-domain.")


if __name__ == "__main__":
    main()
