#!/bin/bash
# h200g : affinage 5 plis (1000 pas, comme f_x1) des deux modèles lancés sans suite : PixelSeal g_x1h (petits trous) et TrustMark g_x1hp (canaux passe-haut)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel11.py"; R=t1-reel
ft () { # filigrane, réserves, tag pré-entraîné, tag affiné, options
  until [ -f $R/runs/$3/$1/fold-1/done ]; do sleep 30; done
  for essai in 1 2 3; do for lot in "0 1 2" "3 4"; do for f in $lot; do [ -f $R/runs/$4/$1/fold$f/done ] || { $PY train --method $1 --suffix $2 --fold $f --tag $4 --init $3 --lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 $5 > $R/logs/$4_$1_f$f.log 2>&1 & sleep 5; }; done; wait; done; done
  $PY agg --tag $4 --write 1 2>&1 | grep -E "^\| (tag|f_).*$1" > $R/logs/agg_$4_$1.txt; }
ft pixelseal _w0,_w1 g_x1h f_x1h "" &
ft trustmark _c0,_c1,_c2,_c3 g_x1hp f_x1hp "--hp 2" &
wait; echo FINI
