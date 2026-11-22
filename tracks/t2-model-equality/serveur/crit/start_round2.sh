#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl
# 1. controle jumeaux deguises, Qwen-1.5B, directions 03 / 28 / 07
setsid nohup $PY -u dir_disg.py --tag F2 --dirs ep-28,ep-03,ep-07 --data $EP --null null_F2.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --bs 16 --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --out x > disg_F2b.log 2>&1 < /dev/null &
# 2. Qwen-3B : directions avec un apprentissage plus fort (rang 16, 6 passes, pas 2e-4), memes fp8 et base
pkill -f "null_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --ep[s]"
sleep 3
$PY - <<'PY'
import json
R = json.load(open("gl_F1.json")); R["dir"] = {}; R["self"] = {}; R["tau"] = {}
json.dump(R, open("gl_F1b.json", "w"))
PY
setsid nohup $PY -u gl_reader.py --data $EP --null null_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --bs 16 --stages dir --epochs 6 --lr 2e-4 --r 16 --scales 0.3,0.6,1.0 --out gl_F1b.json > gl_F1b.log 2>&1 < /dev/null &
sleep 1; echo lances
