# ResMoE: Residual-Corrected MoEfication

Post-hoc modularization of a pretrained LLM's feed-forward (FFN) sub-layers into a
routed mixture-of-experts, **without retraining the base model**. ResMoE improves
each of the three MoEfication processes and shows the third is decisive:

1. **Expert construction → keep-pattern grouping** — cluster neurons by the per-token
   *co-keep* pattern (which neurons are kept together), not by weight similarity.
2. **Expert selection → cheap `x`-router + shared floor** — a small input-conditioned
   router predicts per-group contribution scores; the highest-contribution groups form
   an always-on **shared floor**, the rest are routed under a per-token budget.
3. **Representative → low-rank residual correction (decisive)** — the dropped-expert
   output error `e(t)=Σ_{k∉S}(a_k−ā_k)v_k` is high-rank in *membership* but **low-rank
   and input-predictable** in the *output*; a small (~3% FFN) always-on module predicts
   a rank-`r` correction `ê=B_r ĉ(x)` and adds it back, crossing the selection floor.

We frame this as **software modularization for LLM deployment** and analyze *what
bounds* a decomposition's fidelity (a **routing gap** that is information-bound, and a
**selection floor** that is a rank limit).

## Repository layout

```
experiments/     numbered research scripts (self-contained); key ResMoE scripts below
mple_java/       MultiPL-E Java evaluation harness (needs a JDK on PATH)
ckpts/           fine-tuned checkpoints (git-ignored; regenerate with 251)
```

> `experiments/` also contains earlier exploratory scripts; the ones below produce the
> paper's results.

## Setup

```bash
pip install -r requirements.txt          # Python 3.12, CUDA GPU
# MultiPL-E Java eval additionally needs a JDK (javac/java on PATH).
```

## Key scripts → paper results

| Script | What it produces |
|---|---|
| `experiments/234_relu_causal.py`   | Decoder grid: dense / neuron-oracle / group-oracle / +correction (causal ppl). Main results table (decoders). `KEEPS=0.75,0.50,0.25` |
| `experiments/235_bert_mlm.py`      | Encoder (mBERT/CodeBERT) MLM correction; rank sweep via `RANKS=16,32,64,128` (correction-rank figure) |
| `experiments/248_mbert_modular.py` | mBERT NER/POS fair ablation: co-activation vs keep-pattern grouping vs +correction. `TASK=ner|pos`, `KEEPS=...` |
| `experiments/247_behavior.py`      | Behavior preservation: next-token top-1 agreement + KL vs dense |
| `experiments/249_representative.py`| Representative ablation: zero / mean / conditional-mean / rank-r correction |
| `experiments/246_wallclock.py`     | Realized FFN-sublayer latency (static gather vs dynamic mask) |
| `experiments/251_santacoder_java_ft.py` | Fine-tune base SantaCoder on Java (reproduces G-MoEfication's fine-tuned model) |
| `experiments/243_santacoder_java_corr.py` | SantaCoder MultiPL-E Java: G-MoE baseline vs ResMoE build-up (construction → shared floor → correction), `PPL=1` for perplexity or pass@1 |

### Common environment variables
- `HHMODEL` — HF model id or local checkpoint path
- `KEEPS` — comma-separated keep fractions, e.g. `0.50,0.25`
- `SHARED` — shared-floor fraction of the budget (default `0.5`; ResMoE selection default)
- `PPL=1` — evaluate perplexity instead of the generation benchmark (243)
- `RANKS` — correction-rank sweep (235)

### Example: SantaCoder Java reproduction
```bash
# 1) fine-tune base SantaCoder on Java (matches G-MoEfication's setup)
python3 experiments/251_santacoder_java_ft.py            # -> ckpts/santacoder-java
# 2) G-MoE baseline vs ResMoE (perplexity), keep 85/50
HHMODEL=ckpts/santacoder-java PPL=1 KEEPS=0.85,0.50 \
    python3 experiments/243_santacoder_java_corr.py
```

## Notes
- Oracle / per-neuron references use true activations and are **upper bounds**, not
  deployable; deployable numbers use the `x`-router (+ shared floor) and the
  input-predicted correction.
- Large artifacts (model caches, fine-tuned checkpoints) and third-party papers are
  git-ignored; regenerate checkpoints with `251`.
