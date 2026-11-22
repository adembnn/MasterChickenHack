#!/bin/bash
# h200d : PixelSeal f_s1, relance des plis 0 et 2 tombés faute de mémoire GPU (jusqu'à 3 essais), puis tableau et fichiers (f_s1 seul et moyenne avec le tronc décodeur f_deca)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
for essai in 1 2 3; do
  for f in 0 1 2 3 4; do [ -f $R/runs/f_s1/pixelseal/fold$f/done ] || { $PY train --method pixelseal --suffix _p --fold $f --tag f_s1 --init g_s1 --lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --s1 1 --bs 16 > $R/logs/f_s1_pixelseal_f$f.log 2>&1 & sleep 30; }; done; wait
done
$PY agg --tag f_s1,f_deca,f_s1+f_deca --write 1 2>&1 | grep -E "^\| (tag|f_).*pixelseal" > $R/logs/agg_f_s1_pixelseal.txt; echo FINI
