#!/bin/bash
# Familles des ep : Qwen2.5-1.5B, Llama-3.2-1B, Qwen2.5-3B. LoRA (4 par famille) puis mondes, en série.
cd ~/eac-t2/detecteur
P=.venv/bin/python
SUB=alpaca_r16,gsm_r16,stories_r32,VAL_platypus_r16
for f in qwen1.5 llama1b qwen3; do
  $P lora_train.py --family $f --only $SUB --steps_scale 0.5 >> lora_ep3.log 2>&1
  $P gen.py --family $f --split train --seed 1 --n_base 24 --n_per_variant 2
  $P gen.py --family $f --split val --seed 101 --n_base 16 --n_per_variant 2
done
$P gen.py --family smol360 --split train --seed 2 --n_base 40 --n_per_variant 2
for f in qwen1.5 llama1b qwen3; do $P gen.py --family $f --split train --seed 2 --n_base 24 --n_per_variant 2; done
