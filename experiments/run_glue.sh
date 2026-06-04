#!/bin/bash
cd /home/uoscisai/Experiments/LLM/NNDecomposition
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KEEPS=0.50,0.25
while ! grep -q "MISSING DONE" /tmp/run_missing.log 2>/dev/null; do sleep 20; done
for t in sst2 mrpc qnli; do
  echo "=== GLUE $t ==="
  TASK=$t python3 experiments/241_glue.py > /tmp/241_$t.log 2>&1; echo "$t rc=$?"
done
echo "=== GLUE DONE ==="
