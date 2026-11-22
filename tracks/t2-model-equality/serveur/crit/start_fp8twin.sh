#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl; M=unsloth/Llama-3.2-1B-Instruct
( $PY -u gen_null.py --data $EP --model $M --one ep-05 --temps 0.7,0.8 --k 6 --fp8 e4m3_tensor --out null_F3_fp8.jsonl > fp8twin_F3.log 2>&1 && \
  $PY -u dir_disg.py --tag F3 --groups fp8=null_F3_fp8.jsonl --dirs dz-fp8-fk-0.70-000,dz-fp8-fk-0.80-000,dz-fp8-fk-0.70-001,ep-15,ep-05,ep-11 --data $EP --null null_F3.jsonl --model $M --bs 16 --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --out x >> fp8twin_F3.log 2>&1 ) < /dev/null &
sleep 1; echo lance
