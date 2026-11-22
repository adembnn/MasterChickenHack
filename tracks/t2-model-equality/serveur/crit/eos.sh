#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
run() { setsid nohup $PY -u eos_reader.py --data data50_$1.jsonl --null null50_$1.jsonl --model $2 --tstar q50_$1.json --out eos_$1.json > eos_$1.log 2>&1 < /dev/null & }
run F1 Qwen/Qwen2.5-3B-Instruct
run F2 Qwen/Qwen2.5-1.5B-Instruct
run F3 unsloth/Llama-3.2-1B-Instruct
sleep 1; echo lance-eos
