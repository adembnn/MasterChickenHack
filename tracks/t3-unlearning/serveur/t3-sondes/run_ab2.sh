#!/bin/bash
# 2e lot : reproduction de la reference (meme cible que PG_T85_f20_l48c) puis deux variantes avec partage de probabilite vers le 2e choix d'orig.
export T3ME=$HOME/eac/t3/t3-sondes; P=$HOME/eac/t3/.venv/bin/python; cd $T3ME
while [ ! -f logs/AB_done ]; do sleep 10; done
cp -n cache/tgt_T85_f20_l48.pt cache/tgt_T85_f20_l48_refcopy.pt
for V in T85_f20_l48:AB_ref T85_f20_l48_ru_h20:AB_ru_h20 T85_f20_l48_ru_h35:AB_ru_h35; do
  T=${V%%:*}; N=${V##*:}; cd $T3ME/code
  $P train.py --name $N --fset all --anchor mix --lam 3 --blocks 4 --lr 1e-4 --steps 1500 --tgt $T --w1 1 --wce 1 > $T3ME/logs/$N.log 2>&1
  cd $T3ME && $P mk_from_delta.py $N >> logs/$N.log 2>&1
done
echo FINI > logs/AB2_done
