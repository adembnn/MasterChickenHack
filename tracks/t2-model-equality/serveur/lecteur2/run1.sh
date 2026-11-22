#!/bin/bash
# Point 1 : reproduction du lecteur existant a 50 reponses, trois familles en parallele sur h200h.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/lecteur2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
for TAG in F1 F2 F3; do
case $TAG in
 F1) M=Qwen/Qwen2.5-3B-Instruct; EPS=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33;;
 F2) M=Qwen/Qwen2.5-1.5B-Instruct; EPS=ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32;;
 F3) M=unsloth/Llama-3.2-1B-Instruct; EPS=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31;;
esac
cat collecte_v4_ep_bloc0_20261121T193826Z.jsonl ep_bloc1.jsonl > data50_$TAG.jsonl
cat null_$TAG.jsonl null1_$TAG.jsonl > null50_$TAG.jsonl
setsid nohup $PY -u gl_reader.py --data data50_$TAG.jsonl --null null50_$TAG.jsonl --model $M --eps $EPS --fp8 e4m3_tensor,w8a8_tensor --bs 32 --stages fp8,base,dir --out gl50_$TAG.json > gl50_$TAG.log 2>&1 < /dev/null &
done
sleep 2; echo lance
