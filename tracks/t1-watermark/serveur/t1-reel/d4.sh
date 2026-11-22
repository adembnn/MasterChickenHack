#!/bin/bash
# h200d : cartes de force attendue (4 réserves TrustMark en parallèle), puis 3 variantes à pas égal avec g_s1 (8000 pas) :
#   g_s1e : g_s1 + canal « force attendue » ; g_s2 : aucune réduction sur les deux premiers étages ; PixelSeal g_s1 : U-Net à première couche en pleine résolution
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
C="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --steps 8000"
$PY train --method trustmark --suffix _p,_g1,_g2,_g3 $C --tag g_s2 --s1 2 --bs 8 > $R/logs/g_s2_trustmark.log 2>&1 &
$PY train --method pixelseal --suffix _p $C --tag g_s1 --s1 1 --bs 16 > $R/logs/g_s1_pixelseal.log 2>&1 &
for k in _p _g1 _g2 _g3; do $PY emap --method trustmark --suffix $k > $R/logs/emap${k}_trustmark.log 2>&1 & done
for k in _p _g1 _g2 _g3; do until grep -q "emap ok" $R/logs/emap${k}_trustmark.log 2>/dev/null; do sleep 10; done; done
$PY train --method trustmark --suffix _p,_g1,_g2,_g3 $C --tag g_s1e --s1 1 --bs 16 --echan 1 > $R/logs/g_s1e_trustmark.log 2>&1 &
wait; echo FINI
