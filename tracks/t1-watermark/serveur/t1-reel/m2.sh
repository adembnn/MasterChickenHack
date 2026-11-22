#!/bin/bash
# usage : m2.sh <methode> <reserves> <pas>  -> reprend le pré-entraînement g_x1 au point de contrôle copié, puis affine les 5 plis EN PARALLÈLE (3 essais), tableau et fichiers
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
P="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0"
F="--lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0"
$PY train --method $1 --suffix $2 $P --tag g_x1 --steps $3 >> $R/logs/g_x1_$1.log 2>&1
for essai in 1 2 3; do
  for f in 0 1 2 3 4; do [ -f $R/runs/f_x1/$1/fold$f/done ] || { $PY train --method $1 --suffix $2 --fold $f --tag f_x1 --init g_x1 $F > $R/logs/f_x1_${1}_f$f.log 2>&1 & sleep 5; }; done; wait
done
$PY agg --tag f_x1 --write 1 2>&1 | grep -E "^\| (tag|f_).*$1" > $R/logs/agg_f_x1_$1.txt; echo FINI
