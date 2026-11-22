#!/bin/bash
# h200d : (1) tableau et fichiers PixelSeal maintenant que les 5 plis de f_s1 sont finis ; (2) relance des plis TrustMark f_c1 tombés faute de mémoire GPU, puis tableau
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
$PY agg --tag f_s1,f_deca,f_s1+f_deca --write 1 2>&1 | grep -E "^\| (tag|f_).*pixelseal" > $R/logs/agg_f_s1_pixelseal.txt; cat $R/logs/agg_f_s1_pixelseal.txt | cut -c1-220
F="--lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16"
for f in 0 1 2 3 4; do [ -f $R/runs/f_c1/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _c0,_c1,_c2,_c3 --fold $f --tag f_c1 --init g_c1 $F > $R/logs/f_c1_trustmark_f$f.log 2>&1 & sleep 20; }; done; wait
$PY agg --tag f_c1,f_s1,f_c1+f_s1 --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_f_c1_trustmark.txt; echo FINI
