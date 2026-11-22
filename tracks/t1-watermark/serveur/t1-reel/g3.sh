#!/bin/bash
# h200g (libre) : (1) question « l'affinage est-il assez long ? » : TrustMark f_x1b = affinage 2000 pas avec 8 vraies images sur 16, depuis g_x1 (à comparer à f_x1, CV 0,769)
#                 (2) PixelSeal, le plus faible : encodeur ResNet-50 (g_r50) et tige à filtres passe-haut (g_st), 16 000 pas chacun, puis affinage
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
( for essai in 1 2; do for f in 0 1 2 3 4; do [ -f $R/runs/f_x1b/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _c0,_c1,_c2,_c3 --fold $f --tag f_x1b --init g_x1 --lazy 1 --nreal 8 --steps 2000 --lr 1e-4 --warm 0.1 --ckevery 500 --evalevery 500 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 > $R/logs/f_x1b_trustmark_f$f.log 2>&1 & sleep 5; }; done; wait; done
  $PY agg --tag f_x1,f_x1b,f_x1+f_x1b --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_f_x1b_trustmark.txt ) &
bash $R/c1.sh pixelseal _w0,_w1 16000 r50 "--s1 1 --bs 16 --aux 1.0 --enc resnet50 --seed 2" "" &
bash $R/c1.sh pixelseal _w0,_w1 16000 st "--s1 1 --bs 16 --aux 1.0 --st 1 --hp 2 --seed 3" "" &
wait; echo FINI
