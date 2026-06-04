#!/bin/bash
cd /home/uoscisai/Experiments/LLM/NNDecomposition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CHUNK=512 NCALIB=8192
echo "=== [1/2] Coder-1.5B correction-internals ablation 233 ==="
HHMODEL=Qwen/Qwen2.5-Coder-1.5B python3 experiments/233_corr_ablation.py > /tmp/233_ablation.log 2>&1
echo "=== [1/2] done rc=$? ==="
echo "=== [2/2] Coder-7B HumanEval downstream 232 ==="
HHMODEL=Qwen/Qwen2.5-Coder-7B python3 experiments/232_humaneval_corr.py > /tmp/232_coder7b.log 2>&1
echo "=== [2/2] done rc=$? ==="
echo "=== ALL DONE ==="
