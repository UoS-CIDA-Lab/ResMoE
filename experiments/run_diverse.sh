#!/bin/bash
# Sequential GPU driver (reordered): fully-cached models first, flaky Mistral download last.
cd /home/uoscisai/Experiments/LLM/NNDecomposition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CHUNK=512 NCALIB=8192

echo "=== [1/4] Llama-3.1-8B (SwiGLU, Meta family, 8B) 228 ==="
HHMODEL=meta-llama/Llama-3.1-8B python3 experiments/228_deployable_correction.py > /tmp/228_llama8b.log 2>&1
echo "=== [1/4] done rc=$? ==="

echo "=== [2/4] Qwen2.5-Coder-7B HumanEval downstream 232 ==="
HHMODEL=Qwen/Qwen2.5-Coder-7B python3 experiments/232_humaneval_corr.py > /tmp/232_coder7b.log 2>&1
echo "=== [2/4] done rc=$? ==="

echo "=== [3/4] Qwen2.5-Coder-1.5B correction-internals ablation 233 ==="
HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/233_corr_ablation.py > /tmp/233_ablation.log 2>&1
echo "=== [3/4] done rc=$? ==="

echo "=== [4/4] Mistral-7B-v0.1 (SwiGLU, Mistral family) 228 [xet disabled] ==="
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HHMODEL=mistralai/Mistral-7B-v0.1 python3 experiments/228_deployable_correction.py > /tmp/228_mistral.log 2>&1
echo "=== [4/4] done rc=$? ==="
echo "=== ALL DONE ==="
