#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
( $PY -u make_twin.py > twin_F3.log 2>&1 && $PY -u gl_reader.py --data twin_F3.jsonl --null null_F3.jsonl --model unsloth/Llama-3.2-1B-Instruct --eps tw-01,tw-02,tw-03,tw-04,tw-05,tw-06,tw-07,tw-08,tw-09,tw-10,tw-11 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --stages fp8,base,dir --out gl_twin.json > gl_twin.log 2>&1 ) < /dev/null &
sleep 1; echo lance
