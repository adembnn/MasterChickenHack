#!/bin/bash
# usage: cand3.sh NOM CIBLE_EXACTES CIBLE_VUES [options de train.py]   (h200c) entrainement ou les vues alterees du lot ont leur propre cible (train2.py, TGTV), tableau de bord sur la cible des exactes, fichier, validateur
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; C=~/eac/t3/t3-redistill/tmcode2
n=$1; t=$2; export TGTV=$3; shift 3; cd $T3ME
$C/retry.sh logs/$n.log $PY $C/train2.py --steps 1500 --wce 1 --lam 3 --anchor mix --name $n --tgt $t --fset all "$@" || exit 1
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do $PY $C/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1 && break; sleep 30; done
grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
$PY $C/validate_t3.py candidates/$n.pt --reference ~/eac/t3/model.pt 2>&1 | tail -n 1 >> diag/row_$n.md; sha256sum candidates/$n.pt | cut -c1-16 >> diag/row_$n.md; echo FIN_$n
