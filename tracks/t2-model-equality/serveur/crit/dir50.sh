#!/bin/bash
# Directions a 50 reponses, ciblees : dir50.sh <F1|F2> <groupes de directions, un processus par groupe>
TAG=$1; shift
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
case $TAG in
 F1) M=Qwen/Qwen2.5-3B-Instruct; ALL=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33;;
 F2) M=Qwen/Qwen2.5-1.5B-Instruct; ALL=ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32;;
 F3) M=unsloth/Llama-3.2-1B-Instruct; ALL=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31;;
esac
[ -f data50_$TAG.jsonl ] || cat collecte_v4_ep_bloc0_20261121T193826Z.jsonl ep_bloc1.jsonl > data50_$TAG.jsonl
[ -f null50_$TAG.jsonl ] || cat null_$TAG.jsonl null1_$TAG.jsonl > null50_$TAG.jsonl
k=${K0:-0}; for G in "$@"; do
  rm -f d50_${TAG}_$k.json
  setsid nohup $PY -u gl_reader.py --data data50_$TAG.jsonl --null null50_$TAG.jsonl --model $M --eps $ALL --fp8 "" --bs 16 --kdir 13 --scales 0.3,0.6,1.0 --stages base,dir --fk_base dir --only_dirs $G --out d50_${TAG}_$k.json > d50_${TAG}_$k.log 2>&1 < /dev/null &
  k=$((k + 1))
done
sleep 1; echo "lance dir50 $TAG ($k processus)"
