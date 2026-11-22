#!/bin/bash
# h200f : deux nouvelles réserves d'exemples PixelSeal (autres graines), puis un modèle PixelSeal de plus (20 000 pas, graine 3) pour la moyenne
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel11.py"; R=t1-reel
G="--ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6"
for k in 2 3; do [ -f $R/data_pixelseal_w$k/done ] || $PY gen --method pixelseal --suffix _w$k --gseed 2$k $G --smax 100 > $R/logs/gen_w${k}_pixelseal.log 2>&1 & done; wait; tail -qn 1 $R/logs/gen_w*_pixelseal.log
bash $R/c2.sh pixelseal _w2,_w3 20000 x4 "--s1 1 --bs 16 --aux 1.0 --seed 3" ""
echo FINI
