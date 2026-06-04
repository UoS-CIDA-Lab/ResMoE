#!/bin/bash
cd /home/uoscisai/Experiments/LLM/NNDecomposition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30
echo "=== 248 NER ==="; TASK=ner python3 experiments/248_mbert_modular.py > /tmp/248_ner.log 2>&1; echo "ner rc=$?"
echo "=== 248 POS ==="; TASK=pos python3 experiments/248_mbert_modular.py > /tmp/248_pos.log 2>&1; echo "pos rc=$?"
echo "=== 247 behavioral ==="; python3 experiments/247_behavior.py > /tmp/247_behavior.log 2>&1; echo "247 rc=$?"
echo "=== 243 keep50 ==="; rm -rf mple_java/tmp* 2>/dev/null; KEEPS=0.50 NONEURON=1 python3 experiments/243_santacoder_java_corr.py > /tmp/243_santa.log 2>&1; echo "243 rc=$?"; rm -rf mple_java/tmp* 2>/dev/null
echo "=== 244 phi/falcon ==="; for M in microsoft/phi-2 tiiuae/falcon-7b; do HHMODEL=$M python3 experiments/244_parallel_superglue.py > /tmp/244_$(echo $M|sed 's#/#_#g').log 2>&1; echo "$M rc=$?"; done
echo "=== SEQ DONE ==="
