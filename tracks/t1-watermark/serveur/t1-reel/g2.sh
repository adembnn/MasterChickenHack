#!/bin/bash
# h200g : il ne reste que TrustMark ; dès la fin de son pré-entraînement, affinage des 5 plis EN PARALLÈLE (3 essais), tableau et fichiers
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
F="--lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0"
until [ -f $R/runs/g_x1/trustmark/fold-1/done ]; do sleep 20; done
for essai in 1 2 3; do
  for f in 0 1 2 3 4; do [ -f $R/runs/f_x1/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _c0,_c1,_c2,_c3 --fold $f --tag f_x1 --init g_x1 $F > $R/logs/f_x1_trustmark_f$f.log 2>&1 & sleep 5; }; done; wait
done
$PY agg --tag f_x1 --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_f_x1_trustmark.txt; echo FINI
