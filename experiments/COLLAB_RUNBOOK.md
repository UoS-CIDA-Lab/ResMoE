# Collaborator runbook — T5 MoEfication comparison (exp 254)

Theory-free runbook for running the T5 encoder-decoder MoEfication comparison and
collecting results. You do **not** need to understand ResMoE internals — just run
the commands and check the output has the expected shape.

## What this experiment does (one paragraph)
It takes a pretrained **T5** model (encoder-decoder, ReLU) and, without any
fine-tuning, compares three ways of turning its feed-forward layers into a sparse
"mixture-of-experts", reporting **denoising perplexity** (lower = better). We
expect each successive method to be **better (lower) than the previous one**.

## 0. One-time setup (per server)
```bash
# from the repo root, on a machine with the `resmoe` conda env + a CUDA GPU
conda env list | grep resmoe          # confirm the env exists; if not: pip install -r requirements.txt
git fetch && git checkout feature/t5-moefication
```
Models/data download automatically on first run (needs internet): `t5-base`
(~850MB), `t5-large` (~2.8GB), `t5-3b` (~11GB), and the WikiText corpus.
If a server has no internet, fetch once on a connected machine and copy
`~/.cache/huggingface` over.

## 1. Onboarding smoke test (do this first, on each server)
Run an **existing, known-good** script to confirm the env works and you can read
its output. (This uses a model already cached locally.)
```bash
HHMODEL=Qwen/Qwen2.5-Coder-1.5B conda run -n resmoe python3 experiments/236_baseline_compare.py
```
Good output = four perplexity rows per keep level, each row ≥ the next:
`G-MoE grouping ≥ + keep-pattern ≥ + correction ≥ neuron-oracle`, all ≥ dense.
If two servers print **different** dense numbers for the same model, tell A.

## 2. Run the T5 experiment (the real task)
One model per GPU/server. Logs go to `experiments/results/`.
```bash
mkdir -p experiments/results
# fast smoke (cached small corpus): ~8-12 min on one RTX A6000
WIKI=wikitext-2-raw-v1 HHMODEL=t5-base CUDA_VISIBLE_DEVICES=0 \
  conda run -n resmoe python3 experiments/254_t5_encdec_relu.py \
  2>&1 | tee experiments/results/254_t5-base.log

# full corpus runs (default WIKI=wikitext-103-raw-v1):
HHMODEL=t5-large  CUDA_VISIBLE_DEVICES=1 conda run -n resmoe python3 experiments/254_t5_encdec_relu.py 2>&1 | tee experiments/results/254_t5-large.log
HHMODEL=t5-3b     CUDA_VISIBLE_DEVICES=2 conda run -n resmoe python3 experiments/254_t5_encdec_relu.py 2>&1 | tee experiments/results/254_t5-3b.log
```
Run the three on different GPUs at the same time (they're independent). The
8×A6000 main server alone can run all three; extra servers are a bonus.

Approx runtime per model (1× A6000): t5-base ~8-12 min · t5-large ~25-35 min ·
t5-3b ~1.5-2.5 hr.

## 3. What "good output" looks like
For each keep level the script prints four rows; **each row should be ≤ the one
above it** (lower perplexity), and all ≥ `dense`:
```
  === keep50 (dense X.XXX) ===
    G-MoE/MoEfication grouping (weight k-means + mean)  A
    + keep-pattern grouping (ours grouping)             B   (expect B ≤ A)
    + low-rank correction (OURS full = ResMoE)          C   (expect C ≤ B)
    neuron-oracle (ceiling)                             D   (expect C ≥ D)
```
The gap should be **larger at keep25** than keep50. The first line also prints
`dff`, `nL` (layer count, e.g. enc 12 + dec 12 for t5-base), and calib row counts.

**Flag to A** if: any row is *higher* than the row above it (non-monotonic), a
number is `nan`/`inf`, or the script errors out.

## 4. Record results
Fill this table from the logs (keep50 / keep25 perplexity) and **commit this
runbook** (the raw `.log` files are git-ignored, so the numbers live in this
table — keep the logs locally for reference):

| Model   | corpus | dense | k50 weight | k50 +keeppat | k50 +corr | k50 neuron | k25 weight | k25 +keeppat | k25 +corr | k25 neuron |
|---------|--------|-------|-----------|--------------|-----------|------------|-----------|--------------|-----------|------------|
| t5-base | wt2 (smoke) | 6.346 | 6.902 | 6.386 | 6.336 | 6.376 | 13.173 | 7.786 | 6.422 | 6.944 |
| t5-base | wt103  |       |           |              |           |            |           |              |           |            |
| t5-large| wt103  |       |           |              |           |            |           |              |           |            |
| t5-3b   | wt103  |       |           |              |           |            |           |              |           |            |

(The first row is A's t5-base smoke on `wikitext-2`; redo on `wikitext-103` for the
final table.) Commit the filled table on `feature/t5-moefication`; ping A to review.

## 5. Useful knobs (env vars)
- `HHMODEL` — `t5-base` / `t5-large` / `t5-3b` (must be **non-gated v1.0**; do NOT
  use `t5-v1_1-*` or `flan-t5-*` — the script will assert and stop).
- `KEEPS` — comma list of keep fractions (default `0.50,0.25`).
- `WIKI` — `wikitext-103-raw-v1` (default) or `wikitext-2-raw-v1` (smaller/faster).
- `NCALIB` / `NEVAL` — number of 512-token windows for calibration / eval
  (defaults 48 / 48). Increase for steadier numbers; decrease for a faster smoke.
- `CUDA_VISIBLE_DEVICES` — pin to one GPU so runs don't collide.
