#!/bin/bash
cd /home/uoscisai/Experiments/LLM/NNDecomposition
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CHUNK=512 NCALIB=8192
echo "=== downloading Llama-3.1-8B ==="
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B', allow_patterns=['*.json','*.safetensors','*.model','tokenizer*'])" 2>&1 | tail -1
echo "=== Llama dl done rc=$? ; waiting for 238 to free GPU ==="
while pgrep -f 238_commonsense >/dev/null; do sleep 20; done
echo "=== [1] Mistral-7B-v0.1 228 ==="
HHMODEL=mistralai/Mistral-7B-v0.1 python3 experiments/228_deployable_correction.py > /tmp/228_mistral.log 2>&1; echo "mistral rc=$?"
echo "=== [2] Llama-3.1-8B 228 ==="
HHMODEL=meta-llama/Llama-3.1-8B python3 experiments/228_deployable_correction.py > /tmp/228_llama8b.log 2>&1; echo "llama8b rc=$?"
echo "=== MISSING DONE ==="
