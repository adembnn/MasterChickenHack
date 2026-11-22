#!/bin/bash
# h200d : PixelSeal, affinage 5 plis sur le réel depuis le point de contrôle à 4000 pas du tronc décodeur g_dec (copie figée) ; vraies images non retournées (le décodeur ne lit pas un motif retourné)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
until grep -q '"step": 4000' $R/logs/g_dec_pixelseal.log; do sleep 15; done; sleep 20
mkdir -p $R/runs/g_dec_4k/pixelseal/fold-1 && cp $R/runs/g_dec/pixelseal/fold-1/ckpt.pt $R/runs/g_dec_4k/pixelseal/fold-1/ckpt.pt
for f in 0 1 2 3 4; do
  $PY train --method pixelseal --suffix _p --lazy 1 --fold $f --tag f_deca --init g_dec_4k --arch dec --nreal 6 --raug 0 --tta 1 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 > $R/logs/f_deca_pixelseal_f$f.log 2>&1 &
  [ $((f % 3)) -eq 2 ] && wait
done; wait
$PY agg --tag f_deca --write 1 2>&1 | grep -E "^\| (tag|f_)" > $R/logs/agg_f_deca_pixelseal.txt; cat $R/logs/agg_f_deca_pixelseal.txt; echo FINI
