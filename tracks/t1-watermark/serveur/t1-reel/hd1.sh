#!/bin/bash
# h200d (revenue, disque intact) : reprise aux points de contrôle.
#  1. TrustMark f_s1 : fin de l'affinage 5 plis depuis g_s1 à 8000 pas (plis 2 et 3 repris, pli 4 lancé), tableau et prédictions
#  2. StegaStamp g_s1 (pleine résolution, repris à 1000 pas) jusqu'à 8000, puis affinage 5 plis f_s1
#  3. PixelSeal g_s1 (U-Net pleine résolution, repris à 5000 pas) jusqu'à 8000, puis affinage 5 plis f_s1 et moyenne avec le tronc décodeur f_deca
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
C="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0"; F="--lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --s1 1 --bs 16"
( for f in 2 3 4; do $PY train --method trustmark --suffix _p,_g1,_g2,_g3 --fold $f --tag f_s1 --init g_s1 $F > $R/logs/f_s1_trustmark_f$f.log 2>&1 & done; wait
  $PY agg --tag f_s1,f_s1a,f_s1+f_s1a --write 1 2>&1 | grep -E "^\| (tag|f_)" > $R/logs/agg_f_s1_trustmark.txt ) &
( $PY train --method stegastamp --suffix _p $C --tag g_s1 --s1 1 --bs 16 --steps 8000 >> $R/logs/g_s1_stegastamp.log 2>&1
  for f in 0 1 2 3 4; do $PY train --method stegastamp --suffix _p --fold $f --tag f_s1 --init g_s1 $F > $R/logs/f_s1_stegastamp_f$f.log 2>&1 & [ $((f % 3)) -eq 2 ] && wait; done; wait
  $PY agg --tag f_s1 --write 1 2>&1 | grep -E "^\| (tag|f_).*stegastamp" > $R/logs/agg_f_s1_stegastamp.txt ) &
( $PY train --method pixelseal --suffix _p $C --tag g_s1 --s1 1 --bs 16 --steps 8000 >> $R/logs/g_s1_pixelseal.log 2>&1
  for f in 0 1 2 3 4; do $PY train --method pixelseal --suffix _p --fold $f --tag f_s1 --init g_s1 $F > $R/logs/f_s1_pixelseal_f$f.log 2>&1 & [ $((f % 3)) -eq 2 ] && wait; done; wait
  $PY agg --tag f_s1,f_deca,f_s1+f_deca --write 1 2>&1 | grep -E "^\| (tag|f_).*pixelseal" > $R/logs/agg_f_s1_pixelseal.txt ) &
wait; echo FINI
