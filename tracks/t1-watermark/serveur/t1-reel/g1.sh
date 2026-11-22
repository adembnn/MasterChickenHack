#!/bin/bash
# h200g (carte entière) : la meilleure recette de la nuit pour les trois filigranes :
#   U-Net ResNet-34, première couche en pleine résolution, exemples fabriqués sur la toile réelle (t1-residus), tête d'appoint qui prédit le résidu signé,
#   pré-entraînement long sur fabriqué seul, puis affinage 5 plis sur le réel (relancé jusqu'à 3 fois si un pli tombe), tableau et fichiers de prédiction.
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
$PY cache
G="--ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6"
for k in 0 1 2 3; do $PY gen --method trustmark --suffix _c$k --gseed 1$k $G > $R/logs/gen_c${k}_trustmark.log 2>&1 & done
for k in 0 1; do $PY gen --method stegastamp --suffix _c$k --gseed 1$k $G > $R/logs/gen_c${k}_stegastamp.log 2>&1 & done
for k in 0 1; do $PY gen --method pixelseal --suffix _w$k --gseed 2$k $G --smax 100 > $R/logs/gen_w${k}_pixelseal.log 2>&1 & done
wait; tail -qn 1 $R/logs/gen_*.log
P="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0"
F="--lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0"
chain() {  # chain methode "reserves" pas
  $PY train --method $1 --suffix $2 $P --tag g_x1 --steps $3 > $R/logs/g_x1_$1.log 2>&1
  for essai in 1 2 3; do
    for f in 0 1 2 3 4; do [ -f $R/runs/f_x1/$1/fold$f/done ] || { $PY train --method $1 --suffix $2 --fold $f --tag f_x1 --init g_x1 $F > $R/logs/f_x1_${1}_f$f.log 2>&1 & sleep 5; }; done; wait
  done
  $PY agg --tag f_x1 --write 1 2>&1 | grep -E "^\| (tag|f_).*$1" > $R/logs/agg_f_x1_$1.txt
}
chain stegastamp _c0,_c1 12000 &
chain pixelseal _w0,_w1 12000 &
chain trustmark _c0,_c1,_c2,_c3 16000 &
wait; echo FINI
