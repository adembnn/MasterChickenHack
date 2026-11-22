#!/bin/bash
# h200c : versions en decoupe (entrainees sur a+b, mesurees sur la reserve c) a 50 % : reference t3-methodes, classe redistill, selecteur combine
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; C=~/eac/t3/t3-redistill/tmcode
one() { t=$1; n=PG_${t}cab; cd $T3ME
  $C/retry.sh logs/$n.log $PY $C/train.py --steps 1500 --wce 1 --lam 3 --anchor mix --name $n --tgt $t --fset ab || return 1
  for i in 1 2 3 4 5 6 7 8; do $PY $C/dash.py --fset ab --rset c --tgt $t --tag ab_ --combo $n=$n > logs/dash_$n.log 2>&1 && break; sleep 30; done
  grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md; }
for t in T85_f50_l814 RDc50_l814 RDx50_l814; do one $t & done
wait; echo FIN_DECOUPE
