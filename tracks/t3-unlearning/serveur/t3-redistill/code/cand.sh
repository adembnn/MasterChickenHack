#!/bin/bash
# usage: cand.sh NOM_CIBLE [NOM_CIBLE ...]   (h200c) recette PG du 0,3273 sur des cibles deja fabriquees dans tm/cache : entrainement complet, tableau de bord, fichier, validateur
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; C=~/eac/t3/t3-redistill/tmcode2
one() { t=$1; n=PG_${t}c; cd $T3ME
  $C/retry.sh logs/$n.log $PY $C/train.py --steps 1500 --wce 1 --lam 3 --anchor mix --name $n --tgt $t --fset all || return 1
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do $PY $C/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1 && break; sleep 30; done
  grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
  $PY $C/validate_t3.py candidates/$n.pt --reference ~/eac/t3/model.pt 2>&1 | tail -n 1 >> diag/row_$n.md; sha256sum candidates/$n.pt | cut -c1-16 >> diag/row_$n.md; echo FIN_$n; }
for t in "$@"; do one $t & done; wait; echo FIN_LOT
