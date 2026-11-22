#!/bin/bash
# Point 2a : copies deguisees du jumeau (loi nulle elargie), trois familles en parallele.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/lecteur2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
setsid nohup $PY -u gen_disg2.py --data data50_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --one ep-01 --k 10 --out dg_F1.jsonl > dg_F1.log 2>&1 < /dev/null &
setsid nohup $PY -u gen_disg2.py --data data50_F2.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --one ep-03 --k 10 --out dg_F2.jsonl > dg_F2.log 2>&1 < /dev/null &
setsid nohup $PY -u gen_disg2.py --data data50_F3.jsonl --model unsloth/Llama-3.2-1B-Instruct --one ep-05 --k 10 --out dg_F3.jsonl > dg_F3.log 2>&1 < /dev/null &
sleep 1; echo lance
