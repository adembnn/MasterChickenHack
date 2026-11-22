#!/bin/bash
# h200d : dès que le pré-entraînement TrustMark g_s1 est fini, affinage 5 plis sur les vraies images (2 à la fois), puis tableau et fichiers de prédiction
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel5.py"; R=t1-reel
until [ -f $R/runs/g_s1/trustmark/fold-1/done ]; do sleep 20; done
for f in 0 1 2 3 4; do
  $PY train --method trustmark --suffix _p,_g1,_g2,_g3 --lazy 1 --fold $f --tag f_s1 --init g_s1 --s1 1 --bs 16 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 > $R/logs/f_s1_trustmark_f$f.log 2>&1 &
  [ $((f % 2)) -eq 1 ] && wait
done; wait
$PY agg --tag f_s1 --write 1 2>&1 | grep -E "^\| (tag|f_)" > $R/logs/agg_f_s1_trustmark.txt; cat $R/logs/agg_f_s1_trustmark.txt; echo FINI
