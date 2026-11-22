#!/bin/bash
# usage: spfinish.sh NOM_DELTA NOM_CIBLE   attend l'entrainement en decoupe (fset ab) puis mesure sur la reserve c
cd ~/eac/t3/t3-methodes; n=$1; t=$2
until grep -q "^saved" logs/$n.log 2>/dev/null; do sleep 10; done
for i in 1 2 3 4 5 6 7 8; do ./py code/dash.py --fset ab --rset c --tgt $t --tag ab_ $GRP --combo $n=$n > logs/dash_$n.log 2>&1 && break; sleep 30; done
grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
