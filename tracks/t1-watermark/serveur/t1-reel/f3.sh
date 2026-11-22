#!/bin/bash
# h200f, fin avant 11 h 10 : affinages seuls (5 plis x 1000 pas) depuis le pré-entraînement TrustMark g_x2 (24 000 pas) déjà sur la carte
#  f_x2fh : affinage sur les vraies PLEINES seulement, 90 % de pleines fabriquées (la recette de tmhole, sans pré-entraînement dédié)
#  f_x2s7 : même affinage que f_x2, autre graine (une carte de plus pour la moyenne)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel12.py"; R=t1-reel
ft () { for essai in 1 2 3; do for f in 0 1 2 3 4; do [ -f $R/runs/$1/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _c4,_c5,_c6,_c7 --fold $f --tag $1 --init g_x2 --lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 $2 > $R/logs/$1_trustmark_f$f.log 2>&1 & sleep 3; }; done; wait; done
  $PY agg --tag $1 --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_$1_trustmark.txt; }
ft f_x2fh "--seed 1 --pfull 0.9 --realpop full" &
ft f_x2s7 "--seed 7" &
wait; echo FINI
