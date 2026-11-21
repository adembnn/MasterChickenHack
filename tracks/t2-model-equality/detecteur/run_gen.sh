#!/bin/bash
# LoRA manquantes puis mondes jumeaux, en série (une seule tâche GPU lourde à la fois).
cd ~/eac-t2/detecteur
P=.venv/bin/python
$P gen.py --family smol360 --split val --seed 101 --n_base 24 --n_per_variant 2
$P gen.py --family smol360 --split train --seed 2 --n_base 40 --n_per_variant 2
for f in qwen0.5 smol135 smol1.7; do $P lora_train.py --family $f; done
$P gen.py --family smol360 --split val --seed 102 --n_base 24 --n_per_variant 2
$P gen.py --family qwen0.5 --split train --seed 1 --n_base 30 --n_per_variant 2
$P gen.py --family qwen0.5 --split val --seed 101 --n_base 20 --n_per_variant 2
$P gen.py --family smol135 --split train --seed 1 --n_base 30 --n_per_variant 2
$P gen.py --family smol135 --split val --seed 101 --n_base 20 --n_per_variant 2
$P gen.py --family smol1.7 --split train --seed 1 --n_base 30 --n_per_variant 2
$P gen.py --family smol1.7 --split val --seed 101 --n_base 20 --n_per_variant 2
