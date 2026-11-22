#!/bin/bash
# h200d : toile réelle (mesure de t1-residus) et « voir l'absence de filigrane sur une petite zone ».
#  réserves d'exemples : TrustMark x4, StegaStamp, PixelSeal (toile carrée 512 + s, rognage décentré)
#  TrustMark, pleine résolution, trous fidèles, à pas égal (8000) : g_c1 (référence), g_c1H (+ supervision de la carte 512 x 512 avant regroupement) ; g_c1L : 16 000 pas
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
G="--ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6"
for k in 0 1 2 3; do $PY gen --method trustmark --suffix _c$k --gseed 1$k $G > $R/logs/gen_c${k}_trustmark.log 2>&1 & done
$PY gen --method stegastamp --suffix _c0 --gseed 10 $G > $R/logs/gen_c0_stegastamp.log 2>&1 &
$PY gen --method pixelseal --suffix _c0 --gseed 10 $G > $R/logs/gen_c0_pixelseal.log 2>&1 &
wait; tail -qn 1 $R/logs/gen_c*.log
TM="--method trustmark --suffix _c0,_c1,_c2,_c3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --s1 1 --bs 16 --pshift 0.15 --pblob 0.15"
$PY train $TM --tag smkH --steps 12 --evalevery 12 --ckevery 0 --fullsup 1.0 2>&1 | grep -v -i warn | tail -2 | cut -c1-200; rm -rf $R/runs/smkH
$PY train $TM --tag g_c1 --steps 8000 > $R/logs/g_c1_trustmark.log 2>&1 &
$PY train $TM --tag g_c1H --steps 8000 --fullsup 1.0 > $R/logs/g_c1H_trustmark.log 2>&1 &
$PY train $TM --tag g_c1L --steps 16000 > $R/logs/g_c1L_trustmark.log 2>&1 &
wait; echo FINI
