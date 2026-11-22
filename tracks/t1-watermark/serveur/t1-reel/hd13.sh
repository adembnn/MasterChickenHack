#!/bin/bash
# h200d, TrustMark g_c5cf : perte par paires contrefactuelles de Yassir (PR #34), lambda = 0,1, comparé à g_c5 (lambda = 0) à nombre de pas égal (piste 5)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel10.py"; R=t1-reel
$PY train --method trustmark --suffix _c0,_c1,_c2,_c3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --steps 16000 --aux 1.0 --cf 0.1 --tag g_c5cf > $R/logs/g_c5cf_trustmark.log 2>&1
echo FINI
