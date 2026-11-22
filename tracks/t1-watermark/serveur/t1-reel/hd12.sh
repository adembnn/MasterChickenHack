#!/bin/bash
# h200d, TrustMark, deux questions isolées, chacune comparée à g_c5 (mêmes réserves d'exemples, même graine, même calendrier de 16 000 pas) à nombre de pas égal :
#  g_c5s : --pshift 1.0 (formes de trous déplacées au hasard) au lieu de 0.15 (formes gardées à leur place)  -> que vaut, seule, la correction « trous qui longent les bords » ?
#  g_c5h : --hsmall 0.4 (40 % des trous fabriqués ramenés à 0,5 - 4 % de l'image)                           -> les petits trous réels sont-ils mieux lus ?
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel9.py"; R=t1-reel
P="--method trustmark --suffix _c0,_c1,_c2,_c3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pblob 0.15 --s1 1 --bs 16 --steps 16000 --aux 1.0"
$PY train $P --pshift 1.0 --tag g_c5s > $R/logs/g_c5s_trustmark.log 2>&1 &
$PY train $P --pshift 0.15 --hsmall 0.4 --tag g_c5h > $R/logs/g_c5h_trustmark.log 2>&1 &
wait; echo FINI
