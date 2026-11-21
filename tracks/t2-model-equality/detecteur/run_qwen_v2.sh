#!/bin/bash
# Mondes Qwen SANS repetition_penalty (régénérés). Chaîne A : Qwen-1.5B ; chaîne B : LoRA Qwen-3B puis mondes + relecture des ep.
cd ~/eac-t2/detecteur
P=.venv/bin/python
mkdir -p tainted; for f in worlds/qwen*; do mv $f tainted/ 2>/dev/null; done; mv feats/qwen* tainted/ 2>/dev/null
( for s in "train 5 30 1" "val 101 16 2" "train 8 30 1"; do set -- $s
    $P gen.py --family qwen1.5 --split $1 --seed $2 --n_base $3 --n_per_variant $4; done ) > qwen15_v2.log 2>&1 &
( $P lora_train.py --family qwen3 --only alpaca_r16,gsm_r16,stories_r32,VAL_platypus_r16 --steps_scale 0.3 --bs 2 --accum 8 --ckpt 1 > lora_qwen3c.log 2>&1
  $P feats.py --family qwen3 --inp ep_bloc0_qwen3.jsonl --out feats_ep --tok_budget 1500 > feats_ep_qwen3.log 2>&1
  for s in "train 5 24 1" "val 101 16 2" "train 8 24 1"; do set -- $s
    $P gen.py --family qwen3 --split $1 --seed $2 --n_base $3 --n_per_variant $4; done ) > qwen3_v2.log 2>&1 &
wait
