#!/bin/bash
# Relit chaque monde dès qu'il est complet (marqueur .done), une seule fois.
cd ~/eac-t2/detecteur
while true; do
  for d in worlds/*.jsonl.done; do
    [ -e "$d" ] || continue
    j=${d%.done}; n=$(basename $j .jsonl); fam=${n%%_*}
    [ -e feats/$n.npz ] && continue
    .venv/bin/python feats.py --family $fam --inp $j --out feats 2>&1 | grep -v Loading
  done
  sleep 30
done
