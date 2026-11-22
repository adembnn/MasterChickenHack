#!/bin/bash
# h200h, TrustMark, P2 du critique : perte doublée dans la bande de 2 cases autour de la frontière (--wedge 1.0). Témoin exact : f_r50 (CV 0,762), même modèle de départ, mêmes exemples, 1000 pas.
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel11.py"; R=t1-reel; T=f_r50e
until grep -q FINI $R/logs/h2.log; do sleep 30; done
for essai in 1 2 3; do for f in 0 1 2 3 4; do [ -f $R/runs/$T/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _d0,_d1,_d2,_d3 --fold $f --tag $T --init g_r50 --lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 --enc resnet50 --wedge 1.0 > $R/logs/${T}_trustmark_f$f.log 2>&1 & sleep 5; }; done; wait; done
$PY agg --tag f_r50,$T,f_r50+$T --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_${T}_trustmark.txt; echo FINI
