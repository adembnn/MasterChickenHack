#!/bin/bash
# usage: wave.sh "NOM CIBLE ANCRE" ...   (ANCRE = D : vues degradees + melanges ; M : melanges seuls). Lance, pour chaque triplet, la version complete et la version reserve.
cd ~/eac/t3/t3-methodes; export OMP_NUM_THREADS=4; PYB=${PYB:-./py}
for x in "$@"; do set -- $x; n=$1; t=$2; a=$3
  if [ "$a" = D ]; then A="--anchor deg,mix --lam 3 --lamd 5 --dlo 0.6 --dhi 1.0 --fviews halves"; else A="--anchor mix --lam 3"; fi
  T="$PYB code/train.py $A --steps 1500 --wce 1"
  (nohup code/retry.sh logs/$n.log $T --name $n --tgt $t --fset all > /dev/null 2>&1 < /dev/null &); (nohup code/finish.sh $n $t > /dev/null 2>&1 < /dev/null &)
  (nohup code/retry.sh logs/${n}ab.log $T --name ${n}ab --tgt $t --fset ab > /dev/null 2>&1 < /dev/null &); (nohup code/spfinish.sh ${n}ab $t > /dev/null 2>&1 < /dev/null &)
done
