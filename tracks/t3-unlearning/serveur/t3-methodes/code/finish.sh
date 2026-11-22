#!/bin/bash
# usage: finish.sh NOM_DELTA NOM_CIBLE   attend la fin de l'entrainement, ecrit candidates/NOM.pt, passe tableau de bord et validateur
cd ~/eac/t3/t3-methodes; n=$1; t=$2
until grep -q "^saved" logs/$n.log 2>/dev/null; do sleep 10; done
for i in 1 2 3 4 5 6 7 8; do ./py code/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1 && break; sleep 30; done
grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
./py code/validate_t3.py candidates/$n.pt --reference ../model.pt 2>&1 | tail -n 1 >> diag/row_$n.md; sha256sum candidates/$n.pt | cut -c1-16 >> diag/row_$n.md; ls -l candidates/$n.pt | awk '{print $5}' >> diag/row_$n.md
