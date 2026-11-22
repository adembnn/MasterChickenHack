#!/bin/bash
# h200d : affinage 5 plis de TrustMark g_c5 (16 000 pas avec résidu, autres réserves d'exemples) : un deuxième modèle de la même recette pour la moyenne
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
until [ -f $R/runs/g_c5/trustmark/fold-1/done ]; do sleep 30; done
for essai in 1 2 3; do for f in 0 1 2 3 4; do [ -f $R/runs/f_c5/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _c0,_c1,_c2,_c3 --fold $f --tag f_c5 --init g_c5 --lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 > $R/logs/f_c5_trustmark_f$f.log 2>&1 & sleep 5; }; done; wait; done
$PY agg --tag f_c5 --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_f_c5_trustmark.txt; echo FINI
