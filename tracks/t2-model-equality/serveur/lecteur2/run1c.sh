#!/bin/bash
# Relance du morceau Qwen-3B tombe en memoire (ep-23, ep-24, ep-25), apres la fin des tirages deguises.
cd /home/adem/eac/t2/lecteur2; export HF_HOME=/home/adem/eac/t2/lecteur2/hf; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=/home/adem/eac/t2/.venv/bin/python
until grep -q FINI dg_F1.log && grep -q FINI dg_F2.log && grep -q FINI dg_F3.log; do sleep 20; done
sleep 120
$PY -u gl_reader2.py --data data50_F1.jsonl --null null50_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --fp8 e4m3_tensor --bs 8 --kdir 10 --scales 0.6,1.0,1.5 --stages base,dir --fk_base dir --only_dirs ep-23,ep-24,ep-25 --out g2_F1_p2.json > g2_F1_p2.log 2>&1
