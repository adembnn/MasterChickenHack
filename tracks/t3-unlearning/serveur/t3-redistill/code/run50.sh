#!/bin/bash
# h200c : cibles redistill a 50 % et 60 % (recette et code de t3-methodes copies de h200, loss 8-14), accord de classe fausse, puis 5 candidats
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; C=~/eac/t3/t3-redistill/tmcode; T=~/eac/t3/t3-redistill/targets; POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"
mkdir -p tm/cache tm/logs logs
for f in ~/eac/t3/t3-methodes/cache/*; do b=$(basename $f); case $b in tgt_*|targets_*) ;; *) [ -e tm/cache/$b ] || ln -s $f tm/cache/$b;; esac; done
[ -e tm/cache/tgt_T85_f60_l814.pt ] || cp ~/eac/t3/t3-methodes/cache/tgt_T85_f60_l814.pt tm/cache/
for f in 50 60; do
$PY $C/mk_tgt.py RDc${f}_l814 --pool $POOL --f 0.$f --llo 8 --lhi 14 --rank err --wsrc ext:$T/redistill.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-40
$PY $C/mk_tgt.py RDx${f}_l814 --pool $POOL --f 0.$f --llo 8 --lhi 14 --rank ext:$T/redistill_x_err.npy --wsrc ext:$T/redistill.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-40
done
$PY $C/mk_tgt.py RDh50_l814 --pool $POOL --f 0.50 --llo 8 --lhi 14 --rank err --wsrc ext:$T/redistill.npy --wext 0.5 2>&1 | grep "^RD" | cut -c1-40
$PY code/acc50.py 2>&1 | grep -v -i warn
echo FIN_CIBLES
one() {  # NOM_CIBLE : entrainement complet, tableau de bord, fichier, validateur
  t=$1; n=PG_${t}c; cd $T3ME
  $C/retry.sh logs/$n.log $PY $C/train.py --steps 1500 --wce 1 --lam 3 --anchor mix --name $n --tgt $t --fset all || return 1
  for i in 1 2 3 4 5 6 7 8; do $PY $C/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1 && break; sleep 30; done
  grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
  $PY $C/validate_t3.py candidates/$n.pt --reference ~/eac/t3/model.pt 2>&1 | tail -n 1 >> diag/row_$n.md; sha256sum candidates/$n.pt | cut -c1-16 >> diag/row_$n.md
}
for t in RDc50_l814 RDx50_l814 RDc60_l814 RDx60_l814 RDh50_l814; do one $t & done
wait; echo FIN_CANDIDATS
