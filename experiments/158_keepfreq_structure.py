"""Experiment 158 — WHY does shared+routed help mBERT but not Phi-2 (exp152-157)?
Hypothesis: shared helps iff there is an ALWAYS-ON neuron structure (some neurons kept by the
per-token oracle nearly every token). Hard-wiring those is ~free; if keep-frequency is FLAT
(every neuron kept ~keep-fraction of the time), forcing a static shared set is wasteful.
Measure per-neuron keep-frequency concentration for mBERT (works), Phi-2 & SantaCoder (causal).
Key metric: mean keep-freq of the would-be SHARED set (top 60% of budget by freq). High (->1) =>
free to share (mBERT?); near keep-fraction => wasteful (Phi-2?).
Run: python3 experiments/158_keepfreq_structure.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 4096
SEQ = 512
KEEPS = [0.5, 0.25]
SHARED_FRAC = 0.6
MODELS = [
    ("bert",       "bert-base-multilingual-cased",  "mlm"),
    ("santacoder", "bigcode/gpt_bigcode-santacoder", "causal"),
    ("phi2",       "microsoft/phi-2",               "causal"),
]


def down_module(tag, layer):
    if tag == "bert":
        return layer.output.dense
    if tag == "santacoder":
        return layer.mlp.c_proj
    return layer.mlp.fc2


def layer_list(tag, model):
    if tag == "bert":
        return model.bert.encoder.layer
    return model.model.layers if tag == "phi2" else model.transformer.h


def main():
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
    from datasets import load_dataset
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind in MODELS:
        print(f"\n{'='*58}\n{name}\n{'='*58}", flush=True)
        try:
          tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
          ids = tok(text, return_tensors="pt").input_ids[0][:N_TOK]
          Loader = AutoModelForMaskedLM if kind == "mlm" else AutoModelForCausalLM
          model = Loader.from_pretrained(name, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
          if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
          torch.set_grad_enabled(False)
          layers = layer_list(tag, model); nL = len(layers)
          targets = sorted({nL // 4, nL // 2, 3 * nL // 4})
          cap = {li: [] for li in targets}
          hs = []
          for li in targets:
            dn = down_module(tag, layers[li])
            hs.append(dn.register_forward_pre_hook(
                (lambda li: (lambda _m, a: cap[li].append(a[0].reshape(-1, a[0].shape[-1]).float())))(li)))
          for c0 in range(0, ids.shape[0], SEQ):
            model(ids[c0:c0 + SEQ].unsqueeze(0).to(dev))
          for h in hs:
            h.remove()
          print(f"  layers {nL}, sampled {targets}")
          print(f"  {'keep':>5s} | {'shared-set mean-freq':>20s} | {'frac freq>0.9':>13s} | {'frac freq>0.5':>13s}")
          print("  " + "-" * 60)
          for keep in KEEPS:
            smf, f90, f50 = [], [], []
            for li in targets:
                a = torch.cat(cap[li]).to(dev); dff = a.shape[1]
                dn = down_module(tag, layers[li])
                Wd = dn.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
                abar = a.mean(0); contrib = (a - abar).abs() * Wd.norm(dim=0)
                B = int(round(keep * dff)); T = a.shape[0]
                topB = contrib.argsort(1, descending=True)[:, :B]
                freq = torch.zeros(dff, device=dev)
                freq.scatter_add_(0, topB.reshape(-1), torch.ones(topB.numel(), device=dev)); freq /= T
                n_shared = int(round(SHARED_FRAC * B))
                fs = freq.sort(descending=True).values
                smf.append(fs[:n_shared].mean().item())
                f90.append((freq > 0.9).float().mean().item())
                f50.append((freq > 0.5).float().mean().item())
                del a; gc.collect(); torch.cuda.empty_cache()
            print(f"  {int(keep*100):>4d}% | {np.mean(smf):>19.3f} | {np.mean(f90)*100:>12.1f}% | {np.mean(f50)*100:>12.1f}%", flush=True)
          del model; gc.collect(); torch.cuda.empty_cache()
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: HIGH shared-set mean-freq (->1) & many freq>0.9 neurons => always-on structure")
    print("exists => sharing is ~free (mBERT, shared+routed wins). LOW mean-freq (~keep-frac) &")
    print("few freq>0.9 => flat, no always-on set => forcing shared is wasteful (Phi-2, shared loses).")


if __name__ == "__main__":
    main()
