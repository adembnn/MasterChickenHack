#!/bin/bash
# usage: rdchain.sh NOM_CIBLE all|ab   recette exacte de PG_T85_f30_l814c (t3-methodes) sur une cible redistill, puis tableau de bord (+ fichier complet et validateur si all)
export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; cd $T3ME; t=$1; s=$2; C=~/eac/t3/t3-methodes/code; PY=/home/hermes/eac/kit/.venv/bin/python
if [ $s = all ]; then n=PG_$t; else n=PG_${t}ab; fi
$C/retry.sh logs/$n.log $PY $C/train.py --steps 1500 --wce 1 --lam 3 --anchor mix --name $n --tgt $t --fset $s || exit 1
if [ $s = all ]; then
  for i in 1 2 3 4 5 6 7 8; do $PY $C/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1 && break; sleep 30; done
  grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
  $PY $C/validate_t3.py candidates/$n.pt --reference ~/eac/t3/model.pt 2>&1 | tail -n 1 >> diag/row_$n.md; sha256sum candidates/$n.pt | cut -c1-16 >> diag/row_$n.md
else
  for i in 1 2 3 4 5 6 7 8; do $PY $C/dash.py --fset ab --rset c --tgt $t --tag ab_ --combo $n=$n > logs/dash_$n.log 2>&1 && break; sleep 30; done
  grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
fi
