#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl
nohup $PY -u sys_reader.py --gl gl_F1.json --data $EP --null null_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --out sys_F1.json > sys_F1.log 2>&1 < /dev/null &
nohup $PY -u sys_reader.py --gl gl_F2.json --data $EP --null null_F2.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --out sys_F2.json > sys_F2.log 2>&1 < /dev/null &
nohup $PY -u sys_reader.py --gl gl_F3.json --data $EP --null null_F3.jsonl --model unsloth/Llama-3.2-1B-Instruct --out sys_F3.json > sys_F3.log 2>&1 < /dev/null &
sleep 1; echo lances
