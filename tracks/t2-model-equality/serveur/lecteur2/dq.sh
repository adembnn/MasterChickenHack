#!/bin/bash
# LLR fp8 / int8 / fp16 sur les copies deguisees (contexte sans prompt) : loi nulle elargie des quantifications.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/hf HF_HUB_OFFLINE=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
for i in 1 2 3 4 5 6 7 8 9 10; do
$PY -u fitdisg.py --data data50_$1.jsonl --null dg_$1.jsonl --kplain 0 --kdisg 3 --bs 4 --model $2 --eps $3 --stages C --quant fp8_t,int8_ch,fp16 --out dq_$1.json >> dq_$1.log 2>&1
grep -q FINI dq_$1.log && break; sleep 30; done
