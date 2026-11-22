#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl; RF=collecte_v4_ref_bloc0_20261121T191611Z.jsonl
nohup $PY -u gen_null.py --data $RF --model HuggingFaceTB/SmolLM2-360M-Instruct --one ref-01 --out null_ref.jsonl > null_ref.log 2>&1 < /dev/null &
nohup $PY -u gen_null.py --data $EP --model Qwen/Qwen2.5-3B-Instruct --one ep-01 --out null_F1.jsonl > null_F1.log 2>&1 < /dev/null &
nohup $PY -u gen_null.py --data $EP --model Qwen/Qwen2.5-1.5B-Instruct --one ep-03 --out null_F2.jsonl > null_F2.log 2>&1 < /dev/null &
nohup $PY -u gen_null.py --data $EP --model unsloth/Llama-3.2-1B-Instruct --one ep-05 --out null_F3.jsonl > null_F3.log 2>&1 < /dev/null &
sleep 1; echo lances
