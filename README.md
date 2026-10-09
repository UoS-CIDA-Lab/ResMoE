# ResMoE: Anonymous Reproduction Code

Run all commands from the repository root in **Bash**. This package contains the
experiment entrypoints for the main paper and appendix of “ResMoE: Residual-Corrected MoEfication for Post-Hoc FFN Modularization
of LLMs.” Models and datasets download on first use; a pre-existing
`resmoe` environment, Hugging Face cache, or `ckpts/` directory is not required.
Fine-tuned models are generated explicitly below.

### Install from an empty environment

Use Linux x86_64, an NVIDIA GPU/driver, internet access, and enough disk space for
CUDA wheels, datasets, and checkpoints. The recorded GPU was an RTX A6000 (48 GB).
Start large-model runs on an idle GPU; conversion also needs substantial host RAM.
The final three-seed conversion runs used Python 3.12.13, PyTorch 2.12.1+cu126,
Transformers 5.12.1, and Datasets 5.0.0. The installation commands below pin the
Python packages. Install the **cu126** PyTorch wheel first: the default PyPI wheel
for this PyTorch version uses CUDA 13 and fails on the original CUDA 12.x driver.

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
python -m pip install --no-cache-dir -r /dev/stdin <<'REQUIREMENTS'
transformers==5.12.1
datasets==5.0.0
numpy==2.4.4
matplotlib==3.11.0
seqeval==1.2.2
k-means-constrained==0.7.6
ortools==9.11.4210
lm-eval==0.4.12
accelerate==1.15.0
sentencepiece==0.2.1
protobuf>=4.25,<7
REQUIREMENTS
python -m pip check
mkdir -p experiments/results/reproduction
python -m pip freeze > experiments/results/reproduction/environment.txt
export CUDA_DEVICE_ORDER=PCI_BUS_ID
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
mkdir -p mple_java/lib
curl -fL https://repo.maven.apache.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar \
  -o mple_java/lib/javatuples-1.2.jar
python - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('mple_java/lib/javatuples-1.2.jar').read_bytes()).hexdigest() == \
    '2eda5b19d9820e1cc2f69fcd01639a715a673c11f8507e3d1ed593cf765d5e0a'
PY
python - <<'PY'
from pathlib import Path
from mple_java.runtime import require_java_environment
require_java_environment(Path('mple_java/lib/javatuples-1.2.jar'))
PY
```

The Java setup installs a JDK, including both `javac` and `java`; the required
`mple_java/lib/javatuples-1.2.jar` is downloaded from Maven Central and checked
against its SHA-256 above. Java and HumanEval commands execute generated programs as part of the benchmark. Most experiments
use public models. The Llama row additionally requires access to its gated model
and a Hugging Face token (`HF_TOKEN`) from an account with access.

Numerical equality across packages and hardware is not guaranteed. Where supported,
logs report model revisions, dataset fingerprints, token/source hashes and unrounded
measurements. The synthetic latency experiment measures FFN components rather than
end-to-end inference speed.

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
HHMODEL=google-t5/t5-small WIKI=wikitext-2-raw-v1 NCALIB=2 NEVAL=1 CHUNK=1 MAXROWS=128 KEEPS=0.50 \
  python -u experiments/254_t5_encdec_relu.py \
  2>&1 | tee experiments/results/reproduction/smoke-t5.log
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
  HHMODEL="google-t5/$model" WIKI=wikitext-103-raw-v1 NCALIB=128 NEVAL=64 SEQ=512 CHUNK=8 \
    MAXROWS=16384 NOISE=0.15 MEANSPAN=3 KEEPS=0.75,0.50,0.25 \
    python -u experiments/254_t5_encdec_relu.py \
    2>&1 | tee "experiments/results/reproduction/grid-$model.log"
done
```

Use the full `google-t5/` model IDs: the short `t5-*` aliases can fail with a
404 at the Hugging Face Xet download endpoint in an empty cache.

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
    keep_runs="0.50,0.25"
    if [ "$model" = Qwen/Qwen2.5-7B ]; then
      calibration_device=cpu
      keep_runs="0.50 0.25"
    fi
    for keep_run in $keep_runs; do
      HHMODEL="$model" SEED="$seed" CALIBRATION_DEVICE="$calibration_device" \
        NCALIB=8192 CHUNK=512 KEEPS="$keep_run" \
        python -u experiments/228_deployable_correction.py \
        2>&1 | tee "experiments/results/reproduction/decoder-ranks-${model##*/}-$seed-$keep_run.log"
    done
  done
  for bits in 0 8 4; do
    HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED="$seed" NCALIB=8192 CHUNK=512 \
      KEEPS=0.50 QBITS="$bits" python -u experiments/234_relu_causal.py \
      2>&1 | tee "experiments/results/reproduction/quant-$bits-seed-$seed.log"
  done
done
HHMODEL=Qwen/Qwen2.5-Coder-1.5B SEED=0 KEEPS=0.50,0.25 \
  NEVAL=1024 STEPS=3000 GREF=keep ORACLES_ONLY=1 CORR=0 \
  OUT=experiments/results/reproduction/routing-oracles.json \
  python -u experiments/252_routing_direct_metrics.py \
  2>&1 | tee experiments/results/reproduction/routing-oracles.log
HHMODEL=Qwen/Qwen2.5-Coder-1.5B NCALIB=8192 CHUNK=512 K=128 RFEAT=512 \
  KEEPS=0.50,0.25 TAUS=0.9,0.95,0.99 OUTDIR=experiments/results/reproduction \
  python -u experiments/252_effrank_profile.py \
  2>&1 | tee experiments/results/reproduction/spectrum.log
HHMODEL=Qwen/Qwen2.5-Coder-1.5B KEEPS=0.50,0.25 \
  python -u experiments/247_behavior.py \
  2>&1 | tee experiments/results/reproduction/behavior.log
```

The oracle-only `252` command uses `ORACLES_ONLY=1 CORR=0` and conversion seed 0.
`233` includes SVD/random/reduced-rank-regression bases, MLP/ridge predictors,
and true-coordinate references; keep50/25 are hardcoded. `228` fits 128
coordinates, then truncates for its rank16/32/64/128 comparison. CPU calibration
keeps all 8192 tokens and bounds GPU memory, but changes the SVD device/layout;
use `CALIBRATION_DEVICE=model` for the original Coder rank row and `cpu` for 7B.
The 7B keep50 and keep25 conditions run in separate processes, as in the paper.
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
  HHMODEL=facebook/opt-1.3b SEED="$seed" NCALIB=8192 NEVAL=4096 KEEPS=0.50,0.25,0.10 \
    OUT="experiments/results/reproduction/opt-representative-$seed.json" \
    python -u experiments/214_opt_repzero.py \
    2>&1 | tee "experiments/results/reproduction/opt-representative-$seed.log"
done
for seed in 0 1 2 3 4; do
  HHMODEL=Qwen/Qwen2.5-7B METRIC=superglue SEED="$seed" \
    SHARED=0,0.6 BUDGETS=fixed KEEPS=0.85 \
    NSTRUCT=8192 SWEEP=8192 NEVAL=4096 STEPS=3000 \
    OUT="experiments/results/reproduction/shared-$seed.json" \
    python -u experiments/211_router_sweep.py \
    2>&1 | tee "experiments/results/reproduction/shared-$seed.log"
done
```

`211` adaptive evaluation starts after its 8192 structure and 8192 router-training
tokens. Global allocation is a noncausal reference; deployable thresholds report
actual retained fractions. Shared-expert scores average BoolQ, CB, COPA, RTE, WiC,
and WSC; the command above runs all five recorded conversion seeds.
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
selection uses smaller gathered GEMMs. The full FFN + mask reference computes all
activations and applies one fixed mask shared by every token. These are synthetic
FFN measurements, not end-to-end inference speedups.
`253` measures repeated conversion runs, not a conversion-seed uncertainty study.

### Appendix: router scaling and parallel-block extension

These entrypoints provide the input-router data/capacity experiments, low-rank
routing approximation, controlled grouping comparison, parallel-block extension,
and Java perplexity diagnostic. They retain their original fixed sweeps. All
commands use the environment and output directory created above.

```bash
for script in 208_router_saturation 224_gate_routing 225_xrouter_capacity \
  226_lowrank_gate 236_baseline_compare; do
  HHMODEL=Qwen/Qwen2.5-Coder-1.5B python -u "experiments/$script.py" \
    2>&1 | tee "experiments/results/reproduction/$script.log"
done
HHMODEL=microsoft/phi-2 python -u experiments/244_parallel_superglue.py \
  2>&1 | tee experiments/results/reproduction/parallel-phi2.log
HHMODEL=ckpts/santacoder-java PPL=1 KEEPS=0.85,0.50 \
  python -u experiments/243_santacoder_java_corr.py \
  2>&1 | tee experiments/results/reproduction/java-ppl.log
```

The Java perplexity diagnostic requires the checkpoint generated by `251` above.
`208` varies router-training data size; `225` varies router capacity; `226` uses
weight-SVD routing approximations. These diagnostic protocols are distinct from
the final multi-seed pipeline and its evaluation slices.

### Paper-to-script map

| Paper experiment | Entrypoints |
|---|---|
| Main architecture grid and fake-quantization ablation | `234_relu_causal.py`, `235_bert_mlm.py`, `254_t5_encdec_relu.py` |
| Grouping, matched computation and independent keep sweep | `236_baseline_compare.py`, `254_matched_compute_xrouter.py`, `258_grouping_xrouter.py` |
| Deployable selector/correction pipeline | `230_cheap_select_corr.py` |
| Decoder rank, basis/predictor and representative ablations | `228_deployable_correction.py`, `233_corr_ablation.py`, `249_representative.py` |
| Direct selector and group/neuron oracle diagnostics | `252_routing_direct_metrics.py` |
| Router data/capacity and low-rank routing studies | `208_router_saturation.py`, `224_gate_routing.py`, `225_xrouter_capacity.py`, `226_lowrank_gate.py` |
| HumanEval uncertainty | `257_humaneval_uncertainty.py` |
| Commonsense and GLUE | `238_commonsense.py`, `241_glue.py` |
| Java checkpoint, perplexity and pass@k | `251_santacoder_java_ft.py`, `243_santacoder_java_corr.py`, `256_santacoder_java_passk.py` |
| mBERT NER/POS and parallel-block SuperGLUE | `248_mbert_modular.py`, `244_parallel_superglue.py` |
| Adaptive/shared experts and OPT representatives | `211_router_sweep.py`, `214_opt_repzero.py` |
| Output-error spectra, behavior, conversion cost and synthetic latency | `252_effrank_profile.py`, `247_behavior.py`, `253_conversion_cost.py`, `246_wallclock.py` |

`experiments/baseline_grouping.py` owns normalized constrained input-weight grouping;
`experiments/reproducibility.py` supplies corpus/provenance helpers;
`mple_java/runtime.py` validates Java prerequisites. These are runtime dependencies
of the experiment entrypoints. Models, datasets and generated checkpoints are
obtained by the commands above.
