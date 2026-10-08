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
ckpts/           fine-tuned checkpoints (git-ignored; generate with 251, 256, or 248)
```

> `experiments/` also contains earlier exploratory scripts; the ones below produce the
> paper's results.

## Reproducing the AISTATS 2027 submission

Run all commands from the repository root in **Bash**. This section covers the
submitted main paper and appendix, followed by the earlier experiments retained
in this repository. Models and datasets download on first use; a pre-existing
`resmoe` environment, Hugging Face cache, or `ckpts/` directory is not required.
Fine-tuned models are generated explicitly below.

### Install from an empty environment

Use Linux x86_64, an NVIDIA GPU/driver, internet access, and enough disk space for
CUDA wheels, datasets, and checkpoints. The recorded GPU was an RTX A6000 (48 GB).
Start large-model runs on an idle GPU; conversion also needs substantial host RAM.
The final three-seed conversion runs used Python 3.12.13, PyTorch 2.12.1+cu126,
Transformers 5.12.1, and Datasets 5.0.0. `requirements.txt` pins these Python
packages. Install the **cu126** PyTorch wheel first: the default PyPI wheel for
this PyTorch version uses CUDA 13 and fails on the original CUDA 12.x driver.

If Python/Conda is absent, install [Miniforge](https://github.com/conda-forge/miniforge)
without root privileges (skip this block if `conda` is already available):

```bash
REPRO_TOOLS="$(mktemp -d /tmp/resmoe-tools-XXXXXXXX)"
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o "$REPRO_TOOLS/miniforge.sh"
bash "$REPRO_TOOLS/miniforge.sh" -b -p "$REPRO_TOOLS/miniforge"
export PATH="$REPRO_TOOLS/miniforge/bin:$PATH"
```

Create a new environment and new caches. Keep the printed directory to reactivate
it in another shell; run this block again to restart with an empty environment.

```bash
set -euo pipefail
REPRO_ROOT="$(mktemp -d /tmp/resmoe-reproduce-XXXXXXXX)"
export CONDA_PKGS_DIRS="$REPRO_ROOT/conda-pkgs"
conda create -y --prefix "$REPRO_ROOT/python" --override-channels -c conda-forge \
  python=3.12.13 pip openjdk=17
"$REPRO_ROOT/python/bin/python" -m venv "$REPRO_ROOT/venv"
source "$REPRO_ROOT/venv/bin/activate"
export PATH="$REPRO_ROOT/python/bin:$PATH"
# Keep the venv's python first after adding the JDK.
export PATH="$REPRO_ROOT/venv/bin:$PATH"
export HF_HOME="$REPRO_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$REPRO_ROOT/cache"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r requirements.txt
python -m pip check
mkdir -p experiments/results/reproduction
python -m pip freeze > experiments/results/reproduction/environment.txt
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'Reproduction environment: %s\n' "$REPRO_ROOT"
python - <<'PY'
import torch, transformers, datasets
print(torch.__version__, transformers.__version__, datasets.__version__)
assert torch.cuda.is_available(), 'CUDA is unavailable: check driver and cu126 wheel'
x = torch.ones(2, 2, device='cuda')
assert torch.equal(x @ x, torch.full_like(x, 2))
print(torch.cuda.get_device_name(0))
PY
javac -version
java -version
python - <<'PY'
from pathlib import Path
from mple_java.runtime import require_java_environment
require_java_environment(Path('mple_java/lib/javatuples-1.2.jar'))
PY
```

The Java setup installs a JDK, including both `javac` and `java`; the required
`mple_java/lib/javatuples-1.2.jar` is tracked in this repository. Java and HumanEval
commands execute generated programs as part of the benchmark. Most experiments
use public models. The Llama row additionally requires access to its gated model
and a Hugging Face token (`HF_TOKEN`) from an account with access.

Older retained results used other software versions. In particular, the official
mBERT grouping reruns used PyTorch 2.7.1 / Transformers 5.13.0, and some exploratory
routing results used a different environment. The pinned environment above is the
final three-seed submission environment, not a claim that every historical result
used identical software. Numerical equality across different packages/hardware is
not guaranteed. Scripts that report provenance include model revision, dataset
fingerprints, token/source hashes, and unrounded measurements in their logs.

### First end-to-end checks

These reduced runs test download, calibration, grouping, fitting, and evaluation.
They do **not** reproduce the paper's numerical results. The fitting routines retain
their real optimization loops. Success means exit code zero and finite metrics;
quality rows need not be monotonic or ordered relative to the dense model.

```bash
HHMODEL=facebook/opt-125m NCALIB=512 CHUNK=128 KEEPS=0.50 \
  python -u experiments/234_relu_causal.py \
  2>&1 | tee experiments/results/reproduction/smoke-opt.log
HHMODEL=bert-base-uncased NCALIBSEG=2 NEVALSEG=1 KEEPS=0.50 RANKS=16 \
  python -u experiments/235_bert_mlm.py \
  2>&1 | tee experiments/results/reproduction/smoke-bert.log
HHMODEL=t5-small WIKI=wikitext-2-raw-v1 NCALIB=2 NEVAL=1 CHUNK=1 MAXROWS=128 KEEPS=0.50 \
  python -u experiments/254_t5_encdec_relu.py \
  2>&1 | tee experiments/results/reproduction/smoke-t5.log
```

A small checkpoint-generation check exercises the same training and reload
scripts as the retained T5 downstream experiments (four optimizer steps, not the
paper's fine-tuning recipe):

```bash
HHMODEL=t5-small TASK=mnli MAXTRAIN=16 EPOCHS=1 BS=2 ACCUM=2 \
  OUT=ckpts/reproduction-smoke-t5-small-mnli \
  python -u experiments/256_t5_glue_ft.py \
  2>&1 | tee experiments/results/reproduction/smoke-checkpoint-ft.log
HHMODEL=ckpts/reproduction-smoke-t5-small-mnli TASK=mnli K=64 \
  NCAL=16 NEVAL=8 CHUNK=2 MAXROWS=128 KEEPS=0.25 \
  python -u experiments/257_t5_glue_corr.py \
  2>&1 | tee experiments/results/reproduction/smoke-checkpoint-eval.log
```

### Protocol and outputs

- Causal code experiments concatenate MBPP `full` splits in the order
  `train,test,validation,prompt`; normally the first 8192 tokens are calibration
  and the next 1024 tokens are evaluation. Some earlier/supporting experiments use
  different slices, as specified below. Use the full corpus, not `sanitized`.
- Repeated conversion experiments use `SEED=0,1,2`. Calibration and evaluation
  examples stay fixed; the seed controls the conversion choices described by each
  script. T5's original denoising grid is a separate single-run protocol.
- `KEEPS` is the retained FFN fraction, not measured speedup. `HHMODEL` accepts an
  HF identifier or a local checkpoint path. `NCALIB` counts tokens for causal
  scripts, windows for T5 denoising; encoder MLM uses `NCALIBSEG`/`NEVALSEG`.
- `228`, `230`, `233`, `234`, `235`, and `249` write metrics to stdout, **not** to
  an `OUT` path. Save their logs with `tee`. Other commands below explicitly set
  supported JSON output paths. JSON summary files from prior runs were collected
  from logs; they are not required inputs.
- Retain each seed's raw output. For a reported three-seed mean and uncertainty,
  compute the arithmetic mean and **sample** standard deviation (`ddof=1`) over
  the same row across seeds. Do not average unlike corpus/evaluation protocols.

### Main results: architectures and grouping

Decoder grid (dense / group-oracle / correction / neuron-oracle; keep75/50/25):

```bash
for model in Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B \
  Qwen/Qwen2.5-7B HuggingFaceTB/SmolLM2-1.7B facebook/opt-1.3b; do
  HHMODEL="$model" SEED=0 NCALIB=8192 CHUNK=512 KEEPS=0.75,0.50,0.25 \
    python -u experiments/234_relu_causal.py \
    2>&1 | tee "experiments/results/reproduction/grid-${model##*/}.log"
done
```

The script also supports the retained Mistral/Llama architecture rows:
`HHMODEL=mistralai/Mistral-7B-v0.1` and `HHMODEL=meta-llama/Llama-3.1-8B`, using the
same command/settings. Exact original revisions for these retained rows were not
recorded in the available logs. Large-model calibration may exceed a 48 GB GPU;
use the CPU calibration path in `228` below for the Qwen-7B rank experiment.

Encoder masked-LM grid:

```bash
for model in bert-base-multilingual-cased microsoft/codebert-base-mlm; do
  HHMODEL="$model" SEED=0 NCALIBSEG=256 NEVALSEG=64 KEEPS=0.75,0.50,0.25 \
    python -u experiments/235_bert_mlm.py \
    2>&1 | tee "experiments/results/reproduction/grid-${model##*/}.log"
done
```

T5 v1.0 ReLU encoder-decoder denoising grid (WikiText-103 **test**, without
fine-tuning; these 128/64-window settings differ from the script's smoke defaults):

```bash
for model in t5-small t5-base t5-large t5-3b; do
  HHMODEL="$model" WIKI=wikitext-103-raw-v1 NCALIB=128 NEVAL=64 SEQ=512 CHUNK=8 \
    MAXROWS=16384 NOISE=0.15 MEANSPAN=3 KEEPS=0.75,0.50,0.25 \
    python -u experiments/254_t5_encdec_relu.py \
    2>&1 | tee "experiments/results/reproduction/grid-$model.log"
done
```

Grouping comparison and matched compute: official G-MoEfication **construction**
uses normalized input weights and equal-size constrained k-means. The shared
contribution oracle and PCA/GeLU input router are this repository's controlled
selectors, rather than the published G-MoEfication selector. For SwiGLU, the input
weights are the gate-projection rows. A validated grouping cache is generated on
first use; no prepared group assignments are needed.

```bash
for seed in 0 1 2; do
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" GROUPING_SEED=0 SCOPE=baseline \
    KEEPS=0.50,0.25 NEVAL=1024 \
    GROUPING_DIR=experiments/results/reproduction/weight-groups \
    OUT="experiments/results/reproduction/baseline-$seed.json" \
    python -u experiments/254_matched_compute_xrouter.py \
    2>&1 | tee "experiments/results/reproduction/baseline-$seed.log"
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" GROUPING_SEED=0 SCOPE=pattern_more \
    KEEPS=0.50 NEVAL=1024 OUT="experiments/results/reproduction/matched-$seed.json" \
    python -u experiments/254_matched_compute_xrouter.py \
    2>&1 | tee "experiments/results/reproduction/matched-$seed.log"
  # Each keep starts a separate conversion, matching the submitted keep sweep.
  for keep in 0.85 0.75 0.50 0.25; do
    HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" GROUPING_SEED="$seed" \
      KEEPS="$keep" NEVAL=1024 GROUPS=keep SELS=oracle,xrouter \
      OUT="experiments/results/reproduction/keep-$keep-seed-$seed.json" \
      python -u experiments/258_grouping_xrouter.py \
      2>&1 | tee "experiments/results/reproduction/keep-$keep-seed-$seed.log"
  done
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" \
    python -u experiments/230_cheap_select_corr.py \
    2>&1 | tee "experiments/results/reproduction/pipeline-$seed.log"
done
```

`230` hardcodes keep50/25 and 8192/1024/512 calibration/evaluation/chunk; passing
`KEEPS` does not change it. To compare all constructions under an input router,
run `258` with `GROUPS=weight,coact,keep SELS=xrouter KEEPS=0.50,0.25`.
`254 SCOPE=all` produces the complete original matched-compute diagnostic. Some
retained keep25 matched-compute rows predate the final grouping protocol and
should be treated as historical measurements.

### Downstream tasks and checkpoint generation

HumanEval uncertainty (164 programs, greedy generation, five conversion seeds):

```bash
for model in Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-Coder-7B; do
  HHMODEL="$model" SEEDS=0,1,2,3,4 KEEP=0.50 BATCH=16 MAXNEW=1024 JOBS=8 \
    OUT="experiments/results/reproduction/humaneval-${model##*/}.json" \
    python -u experiments/257_humaneval_uncertainty.py \
    2>&1 | tee "experiments/results/reproduction/humaneval-${model##*/}.log"
done
```

Commonsense uses the paper's three tasks; the script's default adds HellaSwag:

```bash
HHMODEL=Qwen/Qwen2.5-1.5B CALIB=wikitext TASKS=arc_easy,piqa,winogrande \
  python -u experiments/238_commonsense.py \
  2>&1 | tee experiments/results/reproduction/commonsense.log
for task in sst2 mrpc qnli; do
  TASK="$task" NCALIB=4000 NEVAL=100000 KEEPS=0.50,0.25 \
    python -u experiments/241_glue.py \
    2>&1 | tee "experiments/results/reproduction/glue-$task.log"
done
```

`241` downloads the public `textattack/bert-base-uncased-{SST-2,MRPC,QNLI}`
checkpoint for its task; no local fine-tuning artifact is assumed.

SantaCoder Java: train the missing checkpoint first, then measure all rows with
the same MultiPL-E sampling protocol (158 Java problems). The submitted pass@k
comparison uses **keep85**, while keep50 is retained for perplexity.

```bash
HHMODEL=bigcode/gpt_bigcode-santacoder OUT=ckpts/santacoder-java \
  SEQ=512 BS=2 ACCUM=4 LR=1e-5 STEPS=2500 \
  python -u experiments/251_santacoder_java_ft.py \
  2>&1 | tee experiments/results/reproduction/santacoder-ft.log
for seed in 0 1 2; do
  HHMODEL=ckpts/santacoder-java CALIB=mbpp NCALIB=8192 NEVAL=4096 \
    KEEPS=0.85,0.50 PASSK_KEEPS=0.85 ROWS=dense,moef,resmoe SHARED=0.5 \
    N=10 TEMP=0.2 TOP_P=0.95 MAXNEW=350 SEED="$seed" JOBS=16 \
    OUT="experiments/results/reproduction/java-$seed.json" \
    python -u experiments/256_santacoder_java_passk.py \
    2>&1 | tee "experiments/results/reproduction/java-$seed.log"
done
```

Use the native GPTBigCode checkpoint `bigcode/gpt_bigcode-santacoder` shown above.
The Java fine-tuning corpus is `rombodawg/MegaCodeTraining`, extracting Java code
blocks from the `ASSISTANT` field. `256` uses MBPP calibration for the submitted
comparison; `CALIB=java` is an additional in-domain diagnostic. `ROWS` can include
`gmoe,kp,kpsf` for the historical baseline/build-up comparison.

mBERT NER/POS: fine-tune **once per task**, save the dense checkpoint, and reuse
it across conversion seeds. Each result directory must be new.

```bash
for task in ner pos; do
  TASK="$task" FINETUNE_SEED=0 SEED=0 K=64 EPOCHS=2 \
    LANGS=en,de,es,fr,nl,ar,zh,hi,ru KEEPS=0.75,0.50,0.35,0.25,0.20,0.15 \
    RESULT_DIR="experiments/results/reproduction/mbert-$task-seed0" \
    python -u experiments/248_mbert_modular.py \
    2>&1 | tee "experiments/results/reproduction/mbert-$task-seed0.log"
  for seed in 1 2; do
    TASK="$task" FINETUNE_SEED=0 SEED="$seed" K=64 EPOCHS=2 \
      LANGS=en,de,es,fr,nl,ar,zh,hi,ru KEEPS=0.75,0.50,0.35,0.25,0.20,0.15 \
      DENSE_CHECKPOINT="experiments/results/reproduction/mbert-$task-seed0/dense_checkpoint" \
      RESULT_DIR="experiments/results/reproduction/mbert-$task-seed$seed" \
      python -u experiments/248_mbert_modular.py \
      2>&1 | tee "experiments/results/reproduction/mbert-$task-seed$seed.log"
  done
done
```

`248` writes incremental scores, calibration, group assignments, correction
weights, and the checkpoint. This is a pooled nine-language decomposition
comparison: NER entity micro F1 and POS token accuracy, with a contribution
oracle and matched rounded neuron budgets. It does not reproduce the published
42-language protocol or measure sparse execution latency.

### Appendix: diagnostics and ablations

```bash
for seed in 0 1 2; do
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" KEEPS=0.50,0.25 \
    NEVAL=1024 STEPS=3000 GREF=keep CORR=1 \
    OUT="experiments/results/reproduction/routing-$seed.json" \
    python -u experiments/252_routing_direct_metrics.py \
    2>&1 | tee "experiments/results/reproduction/routing-$seed.log"
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" NCALIB=8192 CHUNK=512 \
    python -u experiments/233_corr_ablation.py \
    2>&1 | tee "experiments/results/reproduction/correction-design-$seed.log"
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" KEEPS=0.50,0.25 \
    python -u experiments/249_representative.py \
    2>&1 | tee "experiments/results/reproduction/representatives-$seed.log"
  HHMODEL=bert-base-multilingual-cased SEED="$seed" NCALIBSEG=256 NEVALSEG=64 \
    KEEPS=0.50,0.25 RANKS=16,32,64,128 \
    python -u experiments/235_bert_mlm.py \
    2>&1 | tee "experiments/results/reproduction/mbert-ranks-$seed.log"
  for model in Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-7B; do
    calibration_device=model
    if [ "$model" = Qwen/Qwen2.5-7B ]; then calibration_device=cpu; fi
    HHMODEL="$model" SEED="$seed" CALIBRATION_DEVICE="$calibration_device" \
      NCALIB=8192 CHUNK=512 KEEPS=0.50,0.25 \
      python -u experiments/228_deployable_correction.py \
      2>&1 | tee "experiments/results/reproduction/decoder-ranks-${model##*/}-$seed.log"
  done
  for bits in 0 8 4; do
    HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" NCALIB=8192 CHUNK=512 \
      KEEPS=0.50 QBITS="$bits" python -u experiments/234_relu_causal.py \
      2>&1 | tee "experiments/results/reproduction/quant-$bits-seed-$seed.log"
  done
done
HHMODEL=Qwen/Qwen2.5-Coder-1.5B NCALIB=8192 CHUNK=512 K=128 RFEAT=512 \
  KEEPS=0.50,0.25 TAUS=0.9,0.95,0.99 OUTDIR=experiments/results/reproduction \
  python -u experiments/252_effrank_profile.py \
  2>&1 | tee experiments/results/reproduction/spectrum.log
HHMODEL=Qwen/Qwen2.5-Coder-1.5B KEEPS=0.50,0.25 \
  python -u experiments/247_behavior.py \
  2>&1 | tee experiments/results/reproduction/behavior.log
```

For oracle-only routing diagnostics set `ORACLES_ONLY=1 CORR=0` in `252`.
`233` includes SVD/random/reduced-rank-regression bases, MLP/ridge predictors,
and true-coordinate references; keep50/25 are hardcoded. `228` fits 128
coordinates, then truncates for its rank16/32/64/128 comparison. CPU calibration
keeps all 8192 tokens and bounds GPU memory, but changes the SVD device/layout;
use `CALIBRATION_DEVICE=model` for the original Coder rank row and `cpu` for 7B.
`QBITS=0` is fp16; 8/4 are fake-quantized weights dequantized to fp16 for evaluation,
so the quantization experiment measures quality, not integer-kernel latency.
The spectral diagnostic is separate from the final conversion runs.

Supporting techniques use their own evaluation slices and metrics:

```bash
for seed in 0 1 2; do
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B METRIC=ppl SEED="$seed" \
    SHARED=0.6 BUDGETS=fixed,global,threshold KEEPS=0.85,0.50,0.25 \
    NSTRUCT=8192 SWEEP=8192 NEVAL=4096 STEPS=3000 \
    OUT="experiments/results/reproduction/adaptive-$seed.json" \
    python -u experiments/211_router_sweep.py \
    2>&1 | tee "experiments/results/reproduction/adaptive-$seed.log"
  HHMODEL=Qwen/Qwen2.5-7B METRIC=superglue SEED="$seed" \
    SHARED=0,0.6 BUDGETS=fixed KEEPS=0.85 \
    NSTRUCT=8192 SWEEP=8192 NEVAL=4096 STEPS=3000 \
    OUT="experiments/results/reproduction/shared-$seed.json" \
    python -u experiments/211_router_sweep.py \
    2>&1 | tee "experiments/results/reproduction/shared-$seed.log"
  HHMODEL=facebook/opt-1.3b SEED="$seed" NCALIB=8192 NEVAL=4096 KEEPS=0.50,0.25,0.10 \
    OUT="experiments/results/reproduction/opt-representative-$seed.json" \
    python -u experiments/214_opt_repzero.py \
    2>&1 | tee "experiments/results/reproduction/opt-representative-$seed.log"
done
```

`211` adaptive evaluation starts after its 8192 structure and 8192 router-training
tokens. Global allocation is a noncausal reference; deployable thresholds report
actual retained fractions. Shared-expert scores average BoolQ, CB, COPA, RTE, WiC,
and WSC; additional recorded seeds 3/4 can be run with the same command.
`214` uses WikiText-2 test rather than MBPP, with `K=64`.

Conversion cost and synthetic component latency:

```bash
for model in Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B; do
  for repeat in 0 1 2; do
    HHMODEL="$model" KEEPS=0.50 NCALIB=8192 STEPS=3000 \
      OUT="experiments/results/reproduction/conversion-${model##*/}-$repeat.json" \
      python -u experiments/253_conversion_cost.py \
      2>&1 | tee "experiments/results/reproduction/conversion-${model##*/}-$repeat.log"
  done
done
NTOK=4096 python -u experiments/246_wallclock.py \
  2>&1 | tee experiments/results/reproduction/latency.log
```

`246` downloads no model: random inputs/weights with Qwen dimensions, fp16,
15 warmup and 50 timed iterations, reporting median component latency. Static
selection uses smaller gathered GEMMs; dynamic dense masking still computes the
full FFN. These are synthetic FFN measurements, not end-to-end inference speedups.
`253` measures repeated conversion runs, not a conversion-seed uncertainty study.

### Earlier experiments retained in the repository

The following standalone scripts use their documented fixed sweeps. Capture stdout
with `python -u experiments/<script>.py 2>&1 | tee <log>` after the same setup;
set `HHMODEL` to the model shown. These are additional experiment commands, not
substitutes for the final submission protocols above.

| Experiment | Command / model | Purpose |
|---|---|---|
| 208 | `HHMODEL=Qwen/Qwen2.5-Coder-1.5B python -u experiments/208_router_saturation.py` | Earlier adaptive/router study; historical header still says 178 |
| 215 | `HHMODEL=bigcode/gpt_bigcode-santacoder python -u experiments/215_condmean_gelu_swiglu.py` | Non-ReLU representatives; also Qwen-Coder |
| 216 | `python -u experiments/216_coact_grouping.py` | Co-activation grouping |
| 217 | `python -u experiments/217_balanced_grouping.py` | Balanced grouping |
| 218 | `python -u experiments/218_ksweep_grouping.py` | Expert-count sweep |
| 219 | `python -u experiments/219_spectral_grouping.py` | Spectral grouping |
| 220 | `python -u experiments/220_grouping_blitz.py` | Construction variants |
| 221 | `python -u experiments/221_rank_diagnostic.py` | Error rank diagnostics |
| 222 | `python -u experiments/222_direct_construction.py` | Direct grouping construction |
| 223 | `python -u experiments/223_local_search.py` | Group-local selection search |
| 224 | `python -u experiments/224_gate_routing.py` | Gate/oracle/input-router comparison |
| 225 | `python -u experiments/225_xrouter_capacity.py` | Router capacity and calibration-data sweep |
| 226 | `python -u experiments/226_lowrank_gate.py` | Low-rank gate/up-projection routing |
| 227 | `python -u experiments/227_lowrank_correction.py` | True-error correction ceiling |
| 229 | `python -u experiments/229_end2end.py` | Early end-to-end correction |
| 231 | `python -u experiments/231_corr_vs_moreneurons.py` | Oracle matched-cost diagnostic |
| 232 | `python -u experiments/232_humaneval_corr.py` | Earlier HumanEval oracle rows |
| 236 | `python -u experiments/236_baseline_compare.py` | Earlier weight/co-activation/keep-pattern build-up |
| 237 | `python -u experiments/237_humaneval_baseline.py` | Earlier HumanEval baseline grouping |
| 243 | `HHMODEL=ckpts/santacoder-java PPL=1 KEEPS=0.85,0.50 python -u experiments/243_santacoder_java_corr.py` | Earlier Java-model perplexity; omit `PPL=1` for greedy Java pass@1 |
| 244 | `HHMODEL=microsoft/phi-2 python -u experiments/244_parallel_superglue.py` | Parallel-block SuperGLUE extension |
| 253 budget | `HHMODEL=facebook/opt-1.3b RANKMODE=budget python -u experiments/253_budget_correction.py` | Depth-adaptive rank; repeat with `RANKMODE=fixed` |
| 255 T5 rank | `HHMODEL=t5-large WIKI=wikitext-103-raw-v1 NCALIB=128 NEVAL=64 RANK_KEEP=0.25 RANKS=16,32,64,128 python -u experiments/255_t5_rank_sweep.py` | Earlier T5 correction-rank sweep |

Unless overridden in the table, these scripts default to Qwen-Coder-1.5B. Some
historical sweeps have no environment-variable override for token counts or seeds;
inspect the constants at the top of the script before changing their protocol.

T5 downstream exploration (SST-2 zero-shot; MNLI fine-tuning, separate from the
submitted denoising grid). Expert counts follow `d_ff/32`, and checkpoint names
must be explicitly selected rather than relying on the large-model default:

```bash
for size in small base large 3b; do
  case "$size" in
    small) experts=64; rate=1e-3 ;;
    base) experts=96; rate=1e-3 ;;
    large) experts=128; rate=3e-4 ;;
    3b) experts=512; rate=1e-4 ;;
  esac
  HHMODEL="t5-$size" TASK=sst2 K="$experts" NCAL=4000 NEVAL=0 KEEPS=0.20,0.25 \
    python -u experiments/257_t5_glue_corr.py \
    2>&1 | tee "experiments/results/reproduction/t5-$size-sst2.log"
  HHMODEL="t5-$size" TASK=mnli OPT=adafactor LR="$rate" EPOCHS=3 BS=16 ACCUM=4 \
    OUT="ckpts/t5-$size-mnli" python -u experiments/256_t5_glue_ft.py \
    2>&1 | tee "experiments/results/reproduction/t5-$size-mnli-ft.log"
  HHMODEL="ckpts/t5-$size-mnli" TASK=mnli K="$experts" NCAL=4000 NEVAL=0 KEEPS=0.20,0.25 \
    python -u experiments/257_t5_glue_corr.py \
    2>&1 | tee "experiments/results/reproduction/t5-$size-mnli.log"
done
```

RACE exploration was dropped after fine-tuning failed to match its reference
accuracy. The scripts still accept `TASK=race`; it is not a submitted benchmark.
Use T5 v1.0 (`t5-*`), since gated T5 v1.1/FLAN violates these experiments' ReLU
assumption. Model checkpoints, downloaded data, and results are regenerable and
remain git-ignored.

### Clean validation record (2026-10-07)

Three newly created verification subagents started from source-only copies,
independent Python environments, empty model/dataset caches, and no checkpoints.
The first attempt found version drift in unpinned Transformers (missing
`OPTConfig._commit_hash`); the second found a missing tokenizer asset in the
original tiny-BERT smoke model. Package pins and the documented smoke model were
corrected before restarting from scratch. The third environment installed
Miniforge, Python 3.12.13, OpenJDK 17, CUDA-12.6 PyTorch, and the pinned requirements
successfully. All documented validation commands executed in that final environment
completed with exit code zero and finite metrics.

| Verification | Actual execution |
|---|---|
| Environment and entrypoints | `pip check`, all 47 numbered-script imports, HFLM import, CUDA matrix multiplication, constrained grouping and grouping-cache reload |
| OPT / BERT / T5 smoke commands | Public model/data downloads, calibration, grouping, original 3000-step fitting loops, and evaluation |
| T5 checkpoint smoke | Actual four-step fine-tuning, checkpoint save/reload, calibration, correction fitting, and evaluation; model output labels checked |
| Java checkpoint producer (`251`) | Full MegaCodeTraining download/extraction, actual one-step SantaCoder fine-tuning, checkpoint save/reload, changed weights and finite CUDA loss |
| Java evaluator (`256`) | Native public SantaCoder download, MBPP calibration, all 158 Java problems, greedy plus one sampled completion per problem with `MAXNEW=4` |
| Additional conversion diagnostics | `252` OPT spectra and PNG/PT output; `253` Qwen-Coder-0.5B conversion with `NCALIB=512 STEPS=8`; `253` OPT fixed-rank correction with original 3000-step fitting |
| Program scoring | Actual Java/Python success, assertion failure, and syntax/compilation failure cases; compilation using the tracked javatuples JAR |
| NER/POS data | Fresh downloads of all nine languages for both tasks, all three splits, original label order and token/tag alignment |

The shared corpus loader now requires every MBPP split and propagates download
errors. Java evaluators detect a missing JDK/JAR before generation rather than
recording a spurious zero score. Static checking passed for the shared
reproducibility and Java prerequisite modules. The T5 scripts retain two
pre-existing third-party typing diagnostics on Transformers' decorated
`from_pretrained` method; these calls passed the actual checkpoint and CUDA tests.

This is installation and execution-path validation with reduced workloads, not a
rerun of every full-size paper result. NER/POS full fine-tuning, gated-model access,
large-model memory requirements, and the complete multi-seed/full-length benchmark
grid were not validated by these reduced checks. The complete reproduction commands
above retain the paper's budgets and seeds. A four-token Java completion test is
expected to have poor pass@1 and makes no benchmark-quality claim.
