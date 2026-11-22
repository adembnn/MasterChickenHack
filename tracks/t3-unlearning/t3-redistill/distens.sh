#!/bin/bash
# usage: distens.sh NOM CIBLE_POUR_LE_TABLEAU prof1.pt,prof2.pt,... [options de distens.py]   entrainement + tableau de bord + fichier + validateur
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; C=~/eac/t3/t3-redistill/tmcode2
n=$1; t=$2; te=$3; shift 3; cd $T3ME
for i in $(seq 1 40); do $PY ../code/distens.py --name $n --teachers $te "$@" > logs/$n.log 2>&1 && break; grep -q -i -E "out of memory|ALLOC_FAILED|CUDA error" logs/$n.log || exit 1; sleep 30; done
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do $PY $C/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1 && break; sleep 30; done
grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
$PY $C/validate_t3.py candidates/$n.pt --reference ~/eac/t3/model.pt 2>&1 | tail -n 1 >> diag/row_$n.md; sha256sum candidates/$n.pt | cut -c1-16 >> diag/row_$n.md; echo FIN_$n
