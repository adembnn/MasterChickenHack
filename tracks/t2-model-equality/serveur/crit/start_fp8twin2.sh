#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl; M=unsloth/Llama-3.2-1B-Instruct
setsid nohup $PY -u dir_disg.py --tag F3 --groups fp8=null_F3_fp8.jsonl --dirs dz-fp8-fk-0.70-000,dz-fp8-fk-0.80-000,dz-fp8-fk-0.70-001,ep-15,ep-05,ep-11 --data $EP --null null_F3.jsonl --model $M --bs 16 --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --out x > fp8twin_F3.log 2>&1 < /dev/null &
setsid nohup $PY -u sys_reader2.py --gl gl_F1.json --data $EP --null null_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --bs 16 --out sys2_F1.json > sys2_F1.log 2>&1 < /dev/null &
sleep 1; echo lances
