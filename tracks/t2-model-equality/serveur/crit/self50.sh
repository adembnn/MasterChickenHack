#!/bin/bash
# Lecteur propre (2 plis) a 50 reponses sur les seuls guichets en clair : la base est celui dont rien n'est apprenable (gain dans la loi nulle).
# Usage : self50.sh <F1|F2|F3> <guichets separes par des espaces, un processus par groupe entre guillemets>
TAG=$1; shift
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
case $TAG in F1) M=Qwen/Qwen2.5-3B-Instruct;; F2) M=Qwen/Qwen2.5-1.5B-Instruct;; F3) M=unsloth/Llama-3.2-1B-Instruct;; esac
pkill -f "gl50_${TAG}_p[0-9].json"
sleep 3
k=0; for G in "$@"; do
  setsid nohup $PY -u gl_reader.py --data data50_$TAG.jsonl --null null50_$TAG.jsonl --model $M --eps $G --fp8 "" --bs 16 --stages base,self --fk_base dir --kself $([ $k -eq 0 ] && echo 6 || echo 0) --out self50_${TAG}_$k.json > self50_${TAG}_$k.log 2>&1 < /dev/null &
  k=$((k + 1))
done
sleep 1; echo "lance self50 $TAG ($k processus)"
