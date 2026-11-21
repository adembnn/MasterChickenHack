#!/bin/bash
# Une famille : LoRA (4) puis monde d'entraînement, monde de validation, 2e monde d'entraînement.
# Usage : ./run_fam.sh <famille> [pid à attendre]
cd ~/eac-t2/detecteur
P=.venv/bin/python; f=$1
[ -n "$2" ] && while kill -0 $2 2>/dev/null; do sleep 20; done
$P lora_train.py --family $f --only alpaca_r16,gsm_r16,stories_r32,VAL_platypus_r16 --steps_scale 0.5 >> lora_$f.log 2>&1
$P gen.py --family $f --split train --seed 1 --n_base 24 --n_per_variant 2
$P gen.py --family $f --split val --seed 101 --n_base 16 --n_per_variant 2
$P gen.py --family $f --split train --seed 2 --n_base 30 --n_per_variant 1
