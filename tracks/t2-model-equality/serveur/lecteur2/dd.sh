#!/bin/bash
# Directions cles contre les copies deguisees.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/hf HF_HUB_OFFLINE=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
TAG=$1; M=$2; EPS=$3; PAIRS=$4
[ -s dgn_$TAG.jsonl ] || $PY mkdgn.py dg_$TAG.jsonl dgn_$TAG.jsonl 3
for i in 1 2 3 4 5 6 7 8; do
$PY -u dirdisg.py --data data50_$TAG.jsonl --null dgn_$TAG.jsonl --model $M --eps $EPS --bs 4 --scales 0.6,1.0,1.5 --kdir 99 --out dd_$TAG.json --pairs $PAIRS --map dgn_$TAG.jsonl.map.json >> dd_$TAG.log 2>&1
grep -q FINI dd_$TAG.log && break; sleep 30; done
