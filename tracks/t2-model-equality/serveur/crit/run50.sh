#!/bin/bash
# Lecture a 50 reponses par guichet (bloc 0 + bloc 1). Usage : run50.sh <F1|F2|F3|ref> <journal bloc 1>
TAG=$1; B1=$2
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
case $TAG in
 F1) M=Qwen/Qwen2.5-3B-Instruct; B0=collecte_v4_ep_bloc0_20261121T193826Z.jsonl; EPS=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33;;
 F2) M=Qwen/Qwen2.5-1.5B-Instruct; B0=collecte_v4_ep_bloc0_20261121T193826Z.jsonl; EPS=ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32;;
 F3) M=unsloth/Llama-3.2-1B-Instruct; B0=collecte_v4_ep_bloc0_20261121T193826Z.jsonl; EPS=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31;;
 ref) M=HuggingFaceTB/SmolLM2-360M-Instruct; B0=collecte_v4_ref_bloc0_20261121T191611Z.jsonl; EPS=ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09;;
esac
cat $B0 $B1 > data50_$TAG.jsonl
cat null_$TAG.jsonl null1_$TAG.jsonl > null50_$TAG.jsonl
setsid nohup $PY -u gl_reader.py --data data50_$TAG.jsonl --null null50_$TAG.jsonl --model $M --eps $EPS --fp8 e4m3_tensor,w8a8_tensor --bs 32 --stages fp8,base,dir --out gl50_$TAG.json > gl50_$TAG.log 2>&1 < /dev/null &
sleep 1; echo "lance $TAG"
