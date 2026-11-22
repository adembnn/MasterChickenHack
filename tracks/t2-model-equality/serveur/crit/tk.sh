#!/bin/bash
# Faux endpoints « copie a decodage tronque » (top-k 40), puis lecteur fp8 des guichets a decodage tronque et des autres deguises contre cette loi nulle.
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
run() { ( $PY -u gen_null.py --data data50_$1.jsonl --model $2 --one $3 --temps 0.9,1.0,1.1 --k 40 --top_k 40 --seed 11 --bs 250 --out nulltk_$1.jsonl > nulltk_$1.log 2>&1 && \
  $PY -u gl_reader.py --data data50_$1.jsonl --null nulltk_$1.jsonl --model $2 --eps $4 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --stages fp8,base --out qtk_$1.json > qtk_$1.log 2>&1 ) < /dev/null & }
run F3 unsloth/Llama-3.2-1B-Instruct ep-05 ep-27,ep-16,ep-29,ep-31,ep-19
run F1 Qwen/Qwen2.5-3B-Instruct ep-01 ep-14,ep-02,ep-23,ep-25,ep-33
run F2 Qwen/Qwen2.5-1.5B-Instruct ep-03 ep-07,ep-08,ep-20,ep-06,ep-30
sleep 1; echo lance-tk
