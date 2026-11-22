#!/bin/bash
# h200f : Qwen-3B, directions a apprentissage fort, puis faux endpoints du bloc 1 pour Qwen-3B
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl
$PY -c "
import json
R = json.load(open('gl_F1.json')); R['dir'] = {}; R['self'] = {}; R['tau'] = {}
json.dump(R, open('gl_F1b.json', 'w'))"
setsid nohup $PY -u gl_reader.py --data $EP --null null_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --bs 32 --stages dir --epochs 6 --lr 2e-4 --r 16 --scales 0.3,0.6,1.0 --out gl_F1b.json > gl_F1b.log 2>&1 < /dev/null &
setsid nohup $PY -u gen_null.py --data questions_bloc1.jsonl --model Qwen/Qwen2.5-3B-Instruct --one q --out null1_F1.jsonl > null1_F1.log 2>&1 < /dev/null &
sleep 1; echo lance-f
