#!/bin/bash
# h200c : recalage de la loss du test propre (classe redistill, forme de cible de la reference) : bande de loss 9-15,5 a tau 2,4 ; ou tau 2,1 a bande 8-14
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; C=~/eac/t3/t3-redistill/tmcode; T=~/eac/t3/t3-redistill/targets; POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"
$PY code/flat60.py 0.60 2.1 2>&1 | grep -v -i warn | tail -2
$PY $C/mk_tgt.py RDt24_60_l915 --pool $POOL --f 0.60 --llo 9 --lhi 15.5 --rank err --wsrc ext:$T/redistill_tau2.4.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-40
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
one RDt24_60_l915 all & one RDt21_60_l814 all & one RDt24_60_l915 ab & one RDt21_60_l814 ab &
wait; echo FIN_ETALE2
D=~/eac/t3/t3-redistill/tm/candidates
$PY ~/eac/t3/t3-redistill/code/profil.py T85_f60_l814 ~/eac/t3/t3-methodes/candidates/PG_T85_f60_l814c.pt RDt24_60_l814 $D/PG_RDt24_60_l814c.pt RDt24_60_l915 $D/PG_RDt24_60_l915c.pt RDt21_60_l814 $D/PG_RDt21_60_l814c.pt 2>&1 | grep "^|"
echo FIN_PROFIL2
