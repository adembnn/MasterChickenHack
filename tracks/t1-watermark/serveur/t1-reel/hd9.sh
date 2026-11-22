#!/bin/bash
# h200d : moyennes de variantes TrustMark, puis affinage 5 plis depuis le pré-entraînement long g_c1L (16 000 pas, toile réelle) avec relance automatique
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
$PY agg --tag f_c4,f_s1,f_c1,f_c4+f_s1,f_c4+f_c1,f_c4+f_c1+f_s1 --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_ens_trustmark.txt; cat $R/logs/agg_ens_trustmark.txt | cut -c1-230
F="--lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16"
for essai in 1 2 3; do
  for f in 0 1 2 3 4; do [ -f $R/runs/f_c1L/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _c0,_c1,_c2,_c3 --fold $f --tag f_c1L --init g_c1L $F > $R/logs/f_c1L_trustmark_f$f.log 2>&1 & sleep 20; }; done; wait
done
$PY agg --tag f_c1L,f_c4,f_c4+f_c1L,f_c4+f_c1L+f_s1 --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_f_c1L_trustmark.txt; echo FINI
