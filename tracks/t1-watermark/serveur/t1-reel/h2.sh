#!/bin/bash
# h200h : affinage deux fois plus long (2000 pas, 8 vraies images sur 16), qui a donné +0,005 en CV sur f_x1 (0,769 -> 0,774) : même chose pour le ResNet-50 puis pour g_c5
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel10.py"; R=t1-reel
ft () { # tag pré-entraîné, tag affiné, options
  for essai in 1 2 3; do for f in 0 1 2 3 4; do [ -f $R/runs/$2/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _d0,_d1,_d2,_d3 --fold $f --tag $2 --init $1 --lazy 1 --nreal 8 --steps 2000 --lr 1e-4 --warm 0.1 --ckevery 500 --evalevery 500 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 $3 > $R/logs/$2_trustmark_f$f.log 2>&1 & sleep 5; }; done; wait; done
  $PY agg --tag $2 --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_$2_trustmark.txt; }
ft g_r50 f_r50b "--enc resnet50"
until [ -f $R/runs/g_c5/trustmark/fold-1/ema.pt ]; do sleep 20; done; sleep 20
ft g_c5 f_c5b ""
echo FINI
