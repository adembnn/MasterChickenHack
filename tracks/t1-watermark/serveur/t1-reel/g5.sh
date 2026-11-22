#!/bin/bash
# h200g, PixelSeal, P3b du critique : affinage depuis g_x1 avec 60 % des rectangles fabriqués de petit côté < 64 px, dont un quart de bandes (rapport >= 8). Témoin : f_x1 (CV 0,683), même nombre de pas.
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel11.py"; R=t1-reel; T=f_x1r
for essai in 1 2 3; do for lot in "0 1 2" "3 4"; do for f in $lot; do [ -f $R/runs/$T/pixelseal/fold$f/done ] || { $PY train --method pixelseal --suffix _w0,_w1 --fold $f --tag $T --init g_x1 --lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 --rsmall 0.6 --rband 0.25 > $R/logs/${T}_pixelseal_f$f.log 2>&1 & sleep 5; }; done; wait; done; done
$PY agg --tag f_x1,$T,f_x1+$T --write 1 2>&1 | grep -E "^\| (tag|f_).*pixelseal" > $R/logs/agg_${T}_pixelseal.txt; echo FINI
