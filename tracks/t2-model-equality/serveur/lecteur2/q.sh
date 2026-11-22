#!/bin/bash
# Balayage rapide des quantifications simulees, contexte sans prompt systeme, 24 faux simples.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/hf HF_HUB_OFFLINE=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
Q=fp8_t,fp8_ch,int8_ch,int8_t,int4_g128,nf4_b64,fp4_b64,fp16
for i in 1 2 3 4 5 6 7 8 9 10; do
$PY -u fitdisg.py --data data50_$1.jsonl --null null50_$1.jsonl --kplain 6 --kdisg 0 --bs 4 --model $2 --eps $3 --stages C --quant $Q --out q_$1.json >> q_$1.log 2>&1
grep -q FINI q_$1.log && break; sleep 60; done
