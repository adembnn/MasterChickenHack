#!/bin/bash
# h200d -> PixelSeal : deux réserves d'exemples (toile réelle), puis un modèle avec 9 canaux passe-haut en entrée (--hp 2), 20 000 pas, graine 4 :
# sur TrustMark ce canal donne la même CV et +0,045 en réserve ; sur PixelSeal la tige passe-haut apprise gagne sur les partielles. Membre de plus pour la moyenne.
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel11.py"; R=t1-reel
G="--ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6"
for k in 4 5; do [ -f $R/data_pixelseal_w$k/done ] || $PY gen --method pixelseal --suffix _w$k --gseed 2$k $G --smax 100 > $R/logs/gen_w${k}_pixelseal.log 2>&1 & done; wait; tail -qn 1 $R/logs/gen_w*_pixelseal.log
bash $R/c2.sh pixelseal _w4,_w5 20000 x5hp "--s1 1 --bs 16 --aux 1.0 --seed 4 --hp 2" ""
echo FINI
