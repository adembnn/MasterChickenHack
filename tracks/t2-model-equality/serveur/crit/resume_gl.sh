#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl; RF=collecte_v4_ref_bloc0_20261121T191611Z.jsonl
nohup $PY -u gl_reader.py --data $RF --null null_ref.jsonl --model HuggingFaceTB/SmolLM2-360M-Instruct --eps ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --stages base,dir,self --out gl_ref.json >> gl_ref.log 2>&1 < /dev/null &
nohup $PY -u gl_reader.py --data $EP --null null_F2.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --stages base,dir,self --out gl_F2.json >> gl_F2.log 2>&1 < /dev/null &
nohup $PY -u gl_reader.py --data $EP --null null_F3.jsonl --model unsloth/Llama-3.2-1B-Instruct --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --stages base,dir,self --out gl_F3.json >> gl_F3.log 2>&1 < /dev/null &
sleep 1; echo relances
