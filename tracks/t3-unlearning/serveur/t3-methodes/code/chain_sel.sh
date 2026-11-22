#!/bin/bash
# attend la fin de PD_f30 / PDab_f30 puis lance l'experience des selecteurs (30 % de bascules, loss 8-12, ancre vues degradees)
cd ~/eac/t3/t3-methodes; export OMP_NUM_THREADS=4
until grep -q "^saved" logs/PD_f30.log && grep -q "^saved" logs/PDab_f30.log; do sleep 10; done
T="../.venv/bin/python code/train.py --anchor deg,mix --lam 3 --lamd 5 --dlo 0.6 --dhi 1.0 --fviews halves --steps 1500 --wce 1"
for r in err oof rand; do n=PS30_$r; t=S30_$r
  (nohup code/retry.sh logs/$n.log $T --name $n --tgt $t --fset all > /dev/null 2>&1 < /dev/null &); (nohup code/finish.sh $n $t > /dev/null 2>&1 < /dev/null &); done
(nohup code/retry.sh logs/PS30ab_err.log $T --name PS30ab_err --tgt S30_err --fset ab > /dev/null 2>&1 < /dev/null &); (nohup code/spfinish.sh PS30ab_err S30_err > /dev/null 2>&1 < /dev/null &)
