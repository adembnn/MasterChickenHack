#!/bin/bash
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; PY=/home/adem/eac/t2/.venv/bin/python
EP=collecte_v4_ep_bloc0_20261121T193826Z.jsonl
( $PY -u gen_disg.py F3 unsloth/Llama-3.2-1B-Instruct $EP ep-05 > disg_F3.log 2>&1 && $PY -u dir_disg.py --tag F3 --dirs ep-05,ep-09,ep-11,ep-29,ep-16 --data $EP --null null_F3.jsonl --model unsloth/Llama-3.2-1B-Instruct --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --out x >> disg_F3.log 2>&1 ) < /dev/null &
( $PY -u gen_disg.py F2 Qwen/Qwen2.5-1.5B-Instruct $EP ep-03 > disg_F2.log 2>&1 && $PY -u dir_disg.py --tag F2 --dirs ep-03,ep-28,ep-06,ep-07,ep-08 --data $EP --null null_F2.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --out x >> disg_F2.log 2>&1 ) < /dev/null &
( $PY -u gen_disg.py F1 Qwen/Qwen2.5-3B-Instruct $EP ep-01 > disg_F1.log 2>&1 && $PY -u dir_disg.py --tag F1 --dirs ep-17,ep-18,ep-02,ep-23 --data $EP --null null_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --out x >> disg_F1.log 2>&1 ) < /dev/null &
sleep 1; echo lances
