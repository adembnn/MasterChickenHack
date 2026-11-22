#!/bin/bash
# h200g. Chaque entraînement = une question, comparé à g_x1 du même filigrane (mêmes exemples, même graine, même calendrier) à nombre de pas égal.
#  TrustMark g_x1hp : + 9 canaux passe-haut en entrée (image moins sa version floutée, 3 échelles), en plus de la pleine résolution et de la tête résidu (piste 4)
#  PixelSeal g_x1h  : 40 % des trous fabriqués ramenés à 0,5 - 4 % de l'image (piste 1)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel10.py"; R=t1-reel
C="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0"
$PY train --method trustmark --suffix _c0,_c1,_c2,_c3 $C --steps 16000 --hp 2 --tag g_x1hp > $R/logs/g_x1hp_trustmark.log 2>&1 &
$PY train --method pixelseal --suffix _w0,_w1 $C --steps 12000 --hsmall 0.4 --tag g_x1h > $R/logs/g_x1h_pixelseal.log 2>&1 &
wait; echo FINI
