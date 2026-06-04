#!/bin/bash
cd /home/uoscisai/Experiments/LLM/NNDecomposition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30
echo "=== [246] wall-clock ALONE (idle GPU) ==="
python3 experiments/246_wallclock.py > /tmp/246_wallclock.log 2>&1; echo "246 rc=$?"
echo "=== PARALLEL: 243(Java) ∥ 245(mBERT) ∥ downloads ==="
rm -rf mple_java/tmp* 2>/dev/null
python3 experiments/243_santacoder_java_corr.py > /tmp/243_santa.log 2>&1 & P243=$!
python3 experiments/245_mbert_ner.py > /tmp/245_mbert_ner.log 2>&1 & P245=$!
( for M in microsoft/phi-2 tiiuae/falcon-7b; do
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$M', allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.txt'])" >/dev/null 2>&1
    echo "dl $M rc=$?" >> /tmp/par_dl.log
  done; echo "DL DONE" >> /tmp/par_dl.log ) & PDL=$!
wait $P243 $P245 $PDL
echo "=== 243/245/dl done -> 244 SuperGLUE (serial) ==="; rm -rf mple_java/tmp* 2>/dev/null
for M in microsoft/phi-2 tiiuae/falcon-7b; do
  HHMODEL=$M python3 experiments/244_parallel_superglue.py > /tmp/244_$(echo $M|sed 's#/#_#g').log 2>&1; echo "$M rc=$?"
done
echo "=== PAR DONE ==="
