#!/bin/bash
# Un lecteur LoRA par endpoint (entraine sur cet endpoint seul), z croise de tous les endpoints de la famille.
# Usage : self_readers.sh <journal> <modele> <etiquette> <liste d'endpoints de la famille, virgules> <endpoints a tester, espaces>
D=$1; M=$2; TAG=$3; FAM=$4; shift 4
export HF_HOME=/home/adem/eac/t2/hf
PY=/home/adem/eac/t2/.venv/bin/python
cd /home/adem/eac/t2/crit
for e in "$@"; do
  $PY reader_blind.py --data "$D" --model "$M" --train "$e:1.0" --eps "$FAM" --scales 1.0 --epochs 4 --out "self_${TAG}_${e}.json" > "self_${TAG}_${e}.log" 2>&1
  grep "chelle" "self_${TAG}_${e}.log" | tail -1 | sed "s/^/[$e] /" >> "self_${TAG}.txt"
done
echo FINI >> "self_${TAG}.txt"
