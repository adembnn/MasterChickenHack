#!/bin/bash
# Relance des familles F3 puis F1 (manque de memoire au premier lancement), l'une apres l'autre, une fois le lecteur d'arret termine.
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
while pgrep -f "eos_reade[r]" > /dev/null; do sleep 10; done
run() { $PY -u gen_null.py --data data50_$1.jsonl --model $2 --one $3 --temps 0.9,1.0,1.1 --k 40 --top_k 40 --seed 11 --bs 125 --out nulltk_$1.jsonl > nulltk_$1.log 2>&1 && \
  $PY -u gl_reader.py --data data50_$1.jsonl --null nulltk_$1.jsonl --model $2 --eps $4 --fp8 e4m3_tensor,w8a8_tensor --bs 16 --stages fp8,base --out qtk_$1.json > qtk_$1.log 2>&1; }
run F3 unsloth/Llama-3.2-1B-Instruct ep-05 ep-27,ep-16,ep-29,ep-31,ep-19
run F1 Qwen/Qwen2.5-3B-Instruct ep-01 ep-14,ep-02,ep-23,ep-25,ep-33
echo FINI-TK2
