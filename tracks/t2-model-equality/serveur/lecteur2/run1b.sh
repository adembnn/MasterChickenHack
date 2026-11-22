#!/bin/bash
# Point 1 (relance) : lecteur existant, grille T de 0,6 a 1,2 par 0,1, 10 faux par temperature, directions en morceaux paralleles.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/lecteur2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
go() { # famille modele liste_des_guichets morceaux...
  TAG=$1; M=$2; EPS=$3; shift 3; k=0
  for D in "$@"; do
    if [ $k -eq 0 ]; then ST="fp8,base,dir"; else ST="base,dir"; fi
    setsid nohup $PY -u gl_reader2.py --data data50_$TAG.jsonl --null null50_$TAG.jsonl --model $M --eps $EPS --fp8 e4m3_tensor,w8a8_tensor --bs 16 --kdir 10 --scales 0.6,1.0,1.5 --stages $ST --fk_base dir --only_dirs $D --out g2_${TAG}_p$k.json > g2_${TAG}_p$k.log 2>&1 < /dev/null &
    k=$((k + 1))
  done
}
go F1 Qwen/Qwen2.5-3B-Instruct ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 ep-17,ep-33,ep-01,ep-02 ep-18,ep-04,ep-12,ep-14 ep-23,ep-24,ep-25
go F2 Qwen/Qwen2.5-1.5B-Instruct ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 ep-22,ep-30,ep-03,ep-06,ep-07,ep-08 ep-20,ep-21,ep-26,ep-28,ep-32
go F3 unsloth/Llama-3.2-1B-Instruct ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 ep-05,ep-31,ep-09,ep-10,ep-11,ep-13 ep-15,ep-16,ep-19,ep-27,ep-29
sleep 1; echo lance
