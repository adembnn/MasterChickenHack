#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
for d in 1 2 3; do
  setsid nohup $PY -u gl_reader.py --data ref_frais_$d.jsonl --null null_ref.jsonl --model HuggingFaceTB/SmolLM2-360M-Instruct --eps ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --stages fp8,base,dir --out gl_frais_$d.json > gl_frais_$d.log 2>&1 < /dev/null &
done
sleep 1; echo lances
