#!/bin/bash
# Test A/B : meme recette que PG_T85_f20_l48c (t3-methodes), seule la cible change. Lancer : setsid nohup ./run_ab.sh > logs/run_ab.out 2>&1 < /dev/null &
export T3ME=$HOME/eac/t3/t3-sondes; P=$HOME/eac/t3/.venv/bin/python; mkdir -p $T3ME/logs; cd $T3ME/code
for V in ru lowm; do
  $P train.py --name AB_$V --fset all --anchor mix --lam 3 --blocks 4 --lr 1e-4 --steps 1500 --tgt T85_f20_l48_$V --w1 1 --wce 1 > $T3ME/logs/AB_$V.log 2>&1
  cd $T3ME && $P mk_from_delta.py AB_$V >> $T3ME/logs/AB_$V.log 2>&1; cd $T3ME/code
done
echo FINI > $T3ME/logs/AB_done
