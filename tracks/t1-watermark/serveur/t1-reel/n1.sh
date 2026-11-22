#!/bin/bash
# reprise après la panne des nœuds 6 à 10. usage : n1.sh <k1> <k2> "<chaîne A>" "<chaîne B>"
# génère deux réserves d'exemples PixelSeal (toile réelle) _w<k1>, _w<k2>, puis lance les chaînes données (pré-entraînement, affinage 5 plis, tableau)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/adem/eac/t1/.venv/bin/python t1-reel/reel11.py"; R=t1-reel; mkdir -p $R/logs
[ -f $R/cache/REAL_train.npy ] || $PY cache > $R/logs/cache.log 2>&1
G="--ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6"
for k in $1 $2; do [ -f $R/data_pixelseal_w$k/done ] || $PY gen --method pixelseal --suffix _w$k --gseed 2$k $G --smax 100 > $R/logs/gen_w${k}_pixelseal.log 2>&1 & done; wait; tail -qn 1 $R/logs/gen_w${1}_pixelseal.log $R/logs/gen_w${2}_pixelseal.log
eval "$3" &
eval "$4" &
wait; echo FINI
