#!/bin/bash
# Point 2b : lecteur a deguisement ajuste, lance des que les copies deguisees sont tirees.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/lecteur2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
until grep -q FINI dg_F1.log && grep -q FINI dg_F2.log && grep -q FINI dg_F3.log; do sleep 20; done
$PY -u fitdisg.py --data data50_F1.jsonl --null null50_F1.jsonl,dg_F1.jsonl --kplain 8 --kdisg 6 --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --out fd_F1.json > fd_F1.log 2>&1 &
$PY -u fitdisg.py --data data50_F2.jsonl --null null50_F2.jsonl,dg_F2.jsonl --kplain 8 --kdisg 6 --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --out fd_F2.json > fd_F2.log 2>&1 &
$PY -u fitdisg.py --data data50_F3.jsonl --null null50_F3.jsonl,dg_F3.jsonl --kplain 8 --kdisg 6 --model unsloth/Llama-3.2-1B-Instruct --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --out fd_F3.json > fd_F3.log 2>&1 &
wait
echo FINI3
