#!/bin/bash
# Question unique : à pas égal (1000), ajouter 4 images de test pseudo-étiquetées par lot (étiquette = prédiction du modèle du même pli) améliore-t-il la CV ?
# usage : pl1.sh <tag pré-entraîné> <tag maître> <tag élève> <réserves d'exemples> "<options du modèle>"
G=$1; M=$2; T=$3; S=$4; O=$5
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel8.py"; R=t1-reel
for essai in 1 2 3; do for f in 0 1 2 3 4; do [ -f $R/runs/$T/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix $S --fold $f --tag $T --init $G --lazy 1 --nreal 6 --npl 4 --pl $M --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 $O > $R/logs/${T}_trustmark_f$f.log 2>&1 & sleep 5; }; done; wait; done
$PY agg --tag $M,$T,$M+$T --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_${T}_trustmark.txt; echo FINI
