#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
ALL=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
k=1; for S in ep-02,ep-17,ep-25 ep-04,ep-18,ep-33 ep-12,ep-23; do
  rm -f gl50_F1_p$k.json
  setsid nohup $PY -u gl_reader.py --data data50_F1.jsonl --null null50_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --eps $ALL --fp8 e4m3_tensor,w8a8_tensor --bs 8 --stages base,dir --fk_base dir --only_dirs $S --out gl50_F1_p$k.json > gl50_F1_p$k.log 2>&1 < /dev/null &
  k=$((k + 1)); sleep 20
done
echo relance-F1
