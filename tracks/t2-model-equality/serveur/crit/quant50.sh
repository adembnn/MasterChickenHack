#!/bin/bash
# Autres schemas de quantification (par canal, e5m2) a 50 reponses, loi nulle de 200 faux endpoints.
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
run() { setsid nohup $PY -u gl_reader.py --data data50_$1.jsonl --null null50_$1.jsonl --model $2 --eps $3 --fp8 e4m3_ch,w8a8_ch,e5m2_ch,e4m3_tensor,w8a8_tensor --bs 16 --stages fp8,base --out q50_$1.json > q50_$1.log 2>&1 < /dev/null & }
run F1 Qwen/Qwen2.5-3B-Instruct ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
run F2 Qwen/Qwen2.5-1.5B-Instruct ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
run F3 unsloth/Llama-3.2-1B-Instruct ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
sleep 1; echo lance-quant
