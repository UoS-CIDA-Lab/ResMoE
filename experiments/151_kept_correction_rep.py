"""Experiment 151 — Beat G-MoE's STATIC representative: predict the DROPPED contribution from
the KEPT activations a_kept (a free, high-information signal we never used). G-MoE replaces a
dropped group by its constant MEAN; exp107 predicted from raw x (washed). Here predict
Sum_{dropped} r_g  from a_kept (post-gelu kept activations) via a cheap (low-rank) learned map.
a_kept shares the input x with the dropped neurons -> should predict them far better than mean/x.
Cost: r(k+d) extra (small). mBERT + SantaCoder, single mid layer, fixed top-k selection.
Compare FFN rel-L2: mean (G-MoE) | ridge-from-kept (full, upper bound) | lowrank-r from kept.
Run: python3 experiments/151_kept_correction_rep.py
"""
from __future__ import annotations
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F

N_TOK = 12288
SPLIT = 10240
KEEPS = [0.5, 0.25]
RANKS = [16, 64]
MODELS = [
    ("bert",       "bert-base-multilingual-cased",  "mlm"),
    ("santacoder", "bigcode/gpt_bigcode-santacoder", "causal"),
]


def linears(tag, layer):
    if tag == "bert":
        return layer.intermediate.dense, layer.output.dense
    return layer.mlp.c_fc, layer.mlp.c_proj


def layer_list(tag, model):
    return model.encoder.layer if tag == "bert" else model.transformer.h


def ridge(A, B, lam):
    # solve W: A W ~ B  (A:[N,p], B:[N,d]) -> W:[p,d]
    p = A.shape[1]
    G = A.T @ A + lam * torch.eye(p, device=A.device)
    return torch.linalg.solve(G, A.T @ B)


def main():
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    for tag, name, kind in MODELS:
        print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
        try:
          tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
          ids = tok(text, return_tensors="pt").input_ids[0][:N_TOK]
          Loader = AutoModel if kind == "mlm" else AutoModelForCausalLM
          model = Loader.from_pretrained(name, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
          if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
          torch.set_grad_enabled(False)
          layers = layer_list(tag, model); li = len(layers) // 2
          up, dn = linears(tag, layers[li])
          capa = []
          h2 = dn.register_forward_pre_hook(lambda _m, a: capa.append(a[0].reshape(-1, a[0].shape[-1]).float()))
          chk = 512 if tag == "bert" else 1024
          for c0 in range(0, ids.shape[0], chk):
            model(ids[c0:c0 + chk].unsqueeze(0).to(dev))
          h2.remove()
          a = torch.cat(capa).to(dev)[:N_TOK]; dff = a.shape[1]; T = a.shape[0]
          Wd = dn.weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
          abar = a.mean(0)
          y = a @ Wd.T; ynorm = y.norm(dim=1) + 1e-6
          tr, ev = slice(0, SPLIT), slice(SPLIT, T)
          print(f"  layer {li}, dff={dff}")
          print(f"  {'keep':>5s} | {'mean(G-MoE)':>11s} | {'ridge-kept':>10s} | "
                f"{'lr-16':>6s} | {'lr-64':>6s}")
          print("  " + "-" * 52)
          for keep in KEEPS:
            kk = int(round(keep * dff))
            # fixed top-k neurons by mean |contribution| = |a-abar|*||v|| averaged
            score = ((a - abar).abs() * Wd.norm(dim=0)).mean(0)
            keep_idx = score.topk(kk).indices
            drop_mask = torch.ones(dff, dtype=torch.bool, device=dev); drop_mask[keep_idx] = False
            ak = a[:, keep_idx]                                   # kept activations [N,kk]
            # dropped contribution to output (what we must approximate)
            dropped = (a - abar)[:, drop_mask] @ Wd[:, drop_mask].T   # [N,d]
            # baseline kept output (exact) + mean-rep for dropped
            y_kept = a[:, keep_idx] @ Wd[:, keep_idx].T
            mean_drop = (abar[drop_mask] @ Wd[:, drop_mask].T)        # constant vector

            def rel(pred_drop_ev):
                yhat = y_kept[ev] + pred_drop_ev
                return ((y[ev] - yhat).norm(dim=1) / ynorm[ev]).mean().item() * 100

            # mean (G-MoE): dropped replaced by its mean contribution (constant)
            e_mean = rel(mean_drop.unsqueeze(0).expand(ev.stop - ev.start, -1))
            # ridge-from-kept (full): predict dropped from a_kept (centered)
            akc = ak - ak.mean(0)
            W = ridge(akc[tr], dropped[tr], lam=1e-1)
            e_ridge = rel(akc[ev] @ W)
            # low-rank: PCA of kept acts -> r dims -> ridge
            res = {}
            for r in RANKS:
                _, _, Vt = torch.linalg.svd(akc[tr], full_matrices=False)
                Z = akc @ Vt[:r].T
                Wr = ridge(Z[tr], dropped[tr], lam=1e-2)
                res[r] = rel(Z[ev] @ Wr)
            print(f"  {int(keep*100):>4d}% | {e_mean:>10.1f}% | {e_ridge:>9.1f}% | "
                  f"{res[16]:>5.1f}% | {res[64]:>5.1f}%", flush=True)
          del model; gc.collect(); torch.cuda.empty_cache()
        except Exception as ex:
          import traceback; traceback.print_exc(); print(f"  [SKIP {name}] {ex}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print("\nREAD: lr-r << mean => predicting dropped contribution from KEPT activations beats")
    print("G-MoE's static mean rep (cheap, r(k+d) extra) => a better representative. ~equal => no gain.")


if __name__ == "__main__":
    main()
