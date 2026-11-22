#!/bin/bash
# h200d (h200h injoignable) : point 1 allege, une famille par processus, lots petits pour menager T1.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/hf HF_HUB_OFFLINE=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
run() { setsid nohup $PY -u gl_reader2.py --data data50_$1.jsonl --null null50_$1.jsonl --model $2 --eps $3 --fp8 e4m3_tensor --bs 8 --kdir 10 --scales 0.6,1.0,1.5 --stages fp8,base,dir --fk_base dir --out g2_$1.json > g2_$1.log 2>&1 < /dev/null & }
run F1 Qwen/Qwen2.5-3B-Instruct ep-17,ep-33,ep-18,ep-01,ep-12,ep-24,ep-04,ep-02,ep-14,ep-23,ep-25
run F2 Qwen/Qwen2.5-1.5B-Instruct ep-22,ep-26,ep-30,ep-03,ep-28,ep-21,ep-32,ep-06,ep-07,ep-08,ep-20
run F3 unsloth/Llama-3.2-1B-Instruct ep-05,ep-15,ep-31,ep-09,ep-11,ep-10,ep-13,ep-16,ep-19,ep-27,ep-29
sleep 1; echo lance
