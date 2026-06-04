#!/bin/bash
cd /home/uoscisai/Experiments/LLM/NNDecomposition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1
while pgrep -f 245_mbert_ner >/dev/null; do sleep 30; done
echo "=== 245 done; trimmed 243 (keep50) ==="; rm -rf mple_java/tmp* 2>/dev/null
KEEPS=0.50 NONEURON=1 python3 experiments/243_santacoder_java_corr.py > /tmp/243_santa.log 2>&1; echo "243 rc=$?"; rm -rf mple_java/tmp* 2>/dev/null
echo "=== 244 Phi-2 / Falcon SuperGLUE ==="
for M in microsoft/phi-2 tiiuae/falcon-7b; do
  HHMODEL=$M python3 experiments/244_parallel_superglue.py > /tmp/244_$(echo $M|sed 's#/#_#g').log 2>&1; echo "$M rc=$?"
done
echo "=== REST DONE ==="
