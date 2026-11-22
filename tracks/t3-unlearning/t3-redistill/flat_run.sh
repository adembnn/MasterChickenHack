#!/bin/bash
# h200c : classe redistill avec une cible etalee comme tgt_T85_f60_l814 (temperature tau sur les logits hors pli), recette du 0,3009
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; C=~/eac/t3/t3-redistill/tmcode
$PY code/flat60.py 0.60 2.4 2>&1 | grep -v -i warn
one() { t=$1; s=$2; cd $T3ME; if [ $s = all ]; then n=PG_${t}c; else n=PG_${t}cab; fi
  $C/retry.sh logs/$n.log $PY $C/train.py --steps 1500 --wce 1 --lam 3 --anchor mix --name $n --tgt $t --fset $s || return 1
  if [ $s = all ]; then
    for i in 1 2 3 4 5 6 7 8; do $PY $C/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1 && break; sleep 30; done
    grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
    $PY $C/validate_t3.py candidates/$n.pt --reference ~/eac/t3/model.pt 2>&1 | tail -n 1 >> diag/row_$n.md; sha256sum candidates/$n.pt | cut -c1-16 >> diag/row_$n.md
  else
    for i in 1 2 3 4 5 6 7 8; do $PY $C/dash.py --fset ab --rset c --tgt $t --tag ab_ --combo $n=$n > logs/dash_$n.log 2>&1 && break; sleep 30; done
    grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
  fi; }
one RDt24_60_l814 all & one RDt3_60_l814 all & one RDt4_60_l814 all & one RDt24_60_l814 ab &
wait; echo FIN_ETALE
cd ~/eac/t3/t3-redistill; REF=$(ls ~/eac/t3/t3-methodes/candidates/PG_T85_f60_l814c.pt 2>/dev/null)
$PY code/profil.py ${REF:+T85_f60_l814 $REF} RDc60_l814 tm/candidates/PG_RDc60_l814c.pt RDt24_60_l814 tm/candidates/PG_RDt24_60_l814c.pt RDt3_60_l814 tm/candidates/PG_RDt3_60_l814c.pt RDt4_60_l814 tm/candidates/PG_RDt4_60_l814c.pt 2>&1 | grep "^|"
echo FIN_PROFIL
