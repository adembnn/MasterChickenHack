#!/bin/bash
# h200f : TrustMark, même recette (pleine résolution + toile réelle + résidu), autre graine, 24 000 pas, seul sur la carte ; puis affinage 5 plis
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=4; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
G="--ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6"
for k in 4 5 6 7; do $PY gen --method trustmark --suffix _c$k --gseed 1$k $G > $R/logs/gen_c${k}_trustmark.log 2>&1 & done; wait
bash $R/c1.sh trustmark _c4,_c5,_c6,_c7 24000 x2 "--s1 1 --bs 16 --aux 1.0 --seed 1" ""
