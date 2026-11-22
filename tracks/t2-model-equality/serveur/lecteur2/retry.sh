#!/bin/bash
# Relance un lecteur gl_reader2 jusqu'a FINI (il reprend ce qui est deja sauvegarde). Usage : retry.sh F2 modele guichets
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/hf HF_HUB_OFFLINE=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
for i in $(seq 1 30); do
  $PY -u gl_reader2.py --data data50_$1.jsonl --null null50_$1.jsonl --model $2 --eps $3 --fp8 e4m3_tensor --bs 4 --kdir 10 --scales 0.6,1.0,1.5 --stages fp8,base,dir --fk_base dir --out g2_$1.json >> g2_$1.log 2>&1
  grep -q FINI g2_$1.log && break
  sleep 60
done
