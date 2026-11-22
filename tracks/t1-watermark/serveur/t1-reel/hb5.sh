#!/bin/bash
# h200b : mission « trou des pleines » TrustMark. Deux questions, à pas égal avec g_s1 (pleine résolution, 8000 pas, fabriqué seul) :
#   g_s1F : un modèle dédié aux pleines (fabriqué = pleines seulement) fait-il mieux sur le trou ?   g_s1W : une perte dédiée au trou (Lovász de la classe propre) aide-t-elle ?
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=4; R=t1-reel; PY="python $R/reel6.py"
$PY train --method trustmark --suffix _p --lazy 1 --fold -1 --nreal 0 --tag smk --steps 16 --evalevery 16 --test 0 --s1 1 --bs 8 --pfull 1.0 --wclean 1.0 2>&1 | grep -v -i warn | tail -2 | cut -c1-200; rm -rf $R/runs/smk
C="--method trustmark --suffix _p,_g1,_g2,_g3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --s1 1 --bs 16 --steps 8000"
$PY train $C --tag g_s1F --pfull 1.0 > $R/logs/g_s1F_trustmark.log 2>&1 &
$PY train $C --tag g_s1W --wclean 1.0 > $R/logs/g_s1W_trustmark.log 2>&1 &
wait; echo FINI
