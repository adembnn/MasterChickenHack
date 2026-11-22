#!/bin/bash
# h200e, PixelSeal (le plus faible, et celui qui a le plus gagné : f_x2 CV 0,703) :
#  f_x2b  : affinage long (2000 pas, 8 vraies images) depuis g_x2, témoin f_x2
#  x3     : même recette que x2, autre graine (un modèle de plus pour la moyenne)
#  x3r    : x3 + 50 % des rectangles fabriqués de petit côté < 64 px dont un quart de bandes (P3b au niveau du pré-entraînement) ; témoin exact x3 (même graine), à pas égal
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel11.py"; R=t1-reel
( for essai in 1 2 3; do for f in 0 1 2 3 4; do [ -f $R/runs/f_x2b/pixelseal/fold$f/done ] || { $PY train --method pixelseal --suffix _w0,_w1 --fold $f --tag f_x2b --init g_x2 --lazy 1 --nreal 8 --steps 2000 --lr 1e-4 --warm 0.1 --ckevery 500 --evalevery 500 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 --seed 1 > $R/logs/f_x2b_pixelseal_f$f.log 2>&1 & sleep 5; }; done; wait; done
  $PY agg --tag f_x2,f_x2b --write 1 2>&1 | grep -E "^\| (tag|f_).*pixelseal" > $R/logs/agg_f_x2b_pixelseal.txt ) &
bash $R/c2.sh pixelseal _w0,_w1 20000 x3 "--s1 1 --bs 16 --aux 1.0 --seed 2" "" &
bash $R/c2.sh pixelseal _w0,_w1 20000 x3r "--s1 1 --bs 16 --aux 1.0 --seed 2 --rsmall 0.5 --rband 0.25" "" &
wait; echo FINI
