#!/bin/bash
# h200d : copies deguisees (k = 4 par deguisement), puis lecteur ajuste ; une famille par processus.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/hf HF_HUB_OFFLINE=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
chain() {
  TAG=$1; M=$2; ONE=$3; EPS=$4
  [ -s dg_$TAG.jsonl ] || $PY -u gen_disg2.py --data data50_$TAG.jsonl --model $M --one $ONE --k 4 --bs 192 --out dg_$TAG.jsonl > dg_$TAG.log 2>&1
  $PY -u fitdisg.py --data data50_$TAG.jsonl --null null50_$TAG.jsonl,dg_$TAG.jsonl --kplain 6 --kdisg 4 --bs 8 --model $M --eps $EPS --out fd_$TAG.json > fd_$TAG.log 2>&1
}
chain F1 Qwen/Qwen2.5-3B-Instruct ep-01 ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 &
chain F2 Qwen/Qwen2.5-1.5B-Instruct ep-03 ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 &
chain F3 unsloth/Llama-3.2-1B-Instruct ep-05 ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 &
wait; echo FINID2
