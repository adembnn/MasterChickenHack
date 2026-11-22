#!/bin/bash
# Usage : run_gl.sh <etiquette> <modele> <journal> <endpoints>   (attend la fin du tirage de la loi nulle, puis lance le lecteur)
TAG=$1; M=$2; D=$3; EPS=$4
cd /home/adem/eac/t2/crit; export HF_HOME=/home/adem/eac/t2/hf; PY=/home/adem/eac/t2/.venv/bin/python
until grep -q -e FINI -e Error null_$TAG.log; do sleep 15; done
$PY -u gl_reader.py --data $D --null null_$TAG.jsonl --model $M --eps $EPS --fp8 e4m3_tensor,w8a8_tensor --out gl_$TAG.json > gl_$TAG.log 2>&1
