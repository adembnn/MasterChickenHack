#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl
setsid nohup $PY -u sys_reader2.py --gl gl_F2.json --data $EP --null null_F2.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --bs 16 --out sys2_F2.json > sys2_F2.log 2>&1 < /dev/null &
setsid nohup $PY -u sys_reader2.py --gl gl_F3.json --data $EP --null null_F3.jsonl --model unsloth/Llama-3.2-1B-Instruct --bs 16 --out sys2_F3.json > sys2_F3.log 2>&1 < /dev/null &
sleep 1; echo lances
