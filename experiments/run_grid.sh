#!/bin/bash
cd /home/uoscisai/Experiments/LLM/NNDecomposition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30 KEEPS=0.75,0.50,0.25
while ! grep -q "SEQ DONE" /tmp/run_seq.log 2>/dev/null; do sleep 60; done
echo "=== grid: decoders (234) ==="
for M in Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-1.5B HuggingFaceTB/SmolLM2-1.7B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B mistralai/Mistral-7B-v0.1 meta-llama/Llama-3.1-8B facebook/opt-1.3b; do
  echo "--- $M ---"
  HHMODEL=$M python3 experiments/234_relu_causal.py > /tmp/grid_$(echo $M|sed 's#/#_#g').log 2>&1; echo "$M rc=$?"
done
echo "=== grid: encoders (235) ==="
for M in bert-base-multilingual-cased microsoft/codebert-base-mlm; do
  echo "--- $M ---"
  HHMODEL=$M python3 experiments/235_bert_mlm.py > /tmp/grid_$(echo $M|sed 's#/#_#g').log 2>&1; echo "$M rc=$?"
done
echo "=== GRID DONE ==="
