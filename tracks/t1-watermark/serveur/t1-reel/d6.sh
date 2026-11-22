#!/bin/bash
# h200d : affinage 5 plis sur le réel à partir du point de contrôle à 4000 pas de TrustMark g_s1 (copie figée), sans attendre la fin à 8000 pas
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
until grep -q '"step": 4000' $R/logs/g_s1_trustmark.log; do sleep 15; done; sleep 20
mkdir -p $R/runs/g_s1_4k/trustmark/fold-1 && cp $R/runs/g_s1/trustmark/fold-1/ckpt.pt $R/runs/g_s1_4k/trustmark/fold-1/ckpt.pt
for f in 0 1 2 3 4; do
  $PY train --method trustmark --suffix _p,_g1,_g2,_g3 --lazy 1 --fold $f --tag f_s1a --init g_s1_4k --s1 1 --bs 16 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 > $R/logs/f_s1a_trustmark_f$f.log 2>&1 &
  [ $((f % 2)) -eq 1 ] && wait
done; wait
$PY agg --tag f_s1a --write 1 2>&1 | grep -E "^\| (tag|f_)" > $R/logs/agg_f_s1a_trustmark.txt; cat $R/logs/agg_f_s1a_trustmark.txt; echo FINI
