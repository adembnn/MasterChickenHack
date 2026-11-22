#!/bin/bash
# usage : c1.sh <methode> <reserves> <pas> <etiquette> "<options communes>" "<options d'affinage en plus>"
# pré-entraînement g_<etiquette> sur fabriqué seul, puis affinage f_<etiquette> des 5 plis en parallèle (3 essais), tableau et fichiers
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel; M=$1; S=$2; N=$3; T=$4; O=$5; FO=$6
$PY train --method $M --suffix $S --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pshift 0.15 --pblob 0.15 $O --tag g_$T --steps $N >> $R/logs/g_${T}_$M.log 2>&1
for essai in 1 2 3; do
  for f in 0 1 2 3 4; do [ -f $R/runs/f_$T/$M/fold$f/done ] || { $PY train --method $M --suffix $S --fold $f --tag f_$T --init g_$T --lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 $O $FO > $R/logs/f_${T}_${M}_f$f.log 2>&1 & sleep 5; }; done; wait
done
$PY agg --tag f_$T --write 1 2>&1 | grep -E "^\| (tag|f_).*$M" > $R/logs/agg_f_${T}_$M.txt; echo FINI $T $M
