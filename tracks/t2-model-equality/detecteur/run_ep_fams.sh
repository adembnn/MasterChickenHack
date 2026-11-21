#!/bin/bash
# Familles des ep : Qwen2.5-1.5B, Llama-3.2-1B, Qwen2.5-3B. Chaîne A = LoRA, chaîne B = mondes (attend les LoRA).
cd ~/eac-t2/detecteur
P=.venv/bin/python
SUB=alpaca_r16,dolly_r8qv,gsm_r16,stories_r32,code_r8,hh_r16att,VAL_platypus_r16,VAL_norobots_r8
( for f in qwen1.5 llama1b qwen3; do $P lora_train.py --family $f --only $SUB; done ) > lora_ep.log 2>&1 &
waitl() { while [ $(ls -d loras/$1/*/adapter_config.json 2>/dev/null | wc -l) -lt 8 ]; do sleep 30; done; }
while [ ! -e worlds/smol360_val_s101.jsonl.done ]; do sleep 30; done
for f in qwen1.5 llama1b qwen3; do
  waitl $f
  $P gen.py --family $f --split train --seed 1 --n_base 24 --n_per_variant 2
  $P gen.py --family $f --split val --seed 101 --n_base 16 --n_per_variant 2
done
for f in qwen1.5 llama1b qwen3; do $P gen.py --family $f --split train --seed 2 --n_base 24 --n_per_variant 2; done
$P gen.py --family smol360 --split train --seed 2 --n_base 40 --n_per_variant 2
