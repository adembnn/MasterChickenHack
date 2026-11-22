#!/bin/bash
# cibles redistill a 50 % et 60 % (recette de t3-methodes, loss 8-14) + accord de classe fausse avec les cibles de t3-methodes
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm; PY=/home/hermes/eac/kit/.venv/bin/python; T=~/eac/t3/t3-redistill/targets; POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"
sed -e 's/ap.add_argument(.--strat., type=int, default=0)/ap.add_argument("--strat", type=int, default=0); ap.add_argument("--wext", type=float, default=0.5)/' -e 's/mixw = 0.5 \* e + 0.5 \* q;/mixw = a.wext * e + (1 - a.wext) * q;/' ~/eac/t3/t3-methodes/code/mk_tgt.py > code/mk_tgt_rd.py
echo "wext dans mk_tgt_rd.py : $(grep -c 'a.wext' code/mk_tgt_rd.py)"
$PY ~/eac/t3/t3-methodes/code/mk_tgt.py T85_f60_l814 --pool $POOL --f 0.60 --llo 8 --lhi 14 2>&1 | grep "^T85" | cut -c1-60
for f in 50 60; do
$PY code/mk_tgt_rd.py RDc${f}_l814 --pool $POOL --f 0.$f --llo 8 --lhi 14 --rank err --wsrc ext:$T/redistill.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-40
$PY code/mk_tgt_rd.py RDx${f}_l814 --pool $POOL --f 0.$f --llo 8 --lhi 14 --rank ext:$T/redistill_x_err.npy --wsrc ext:$T/redistill.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-40
done
$PY code/mk_tgt_rd.py RDh50_l814 --pool $POOL --f 0.50 --llo 8 --lhi 14 --rank err --wsrc ext:$T/redistill.npy --wext 0.5 2>&1 | grep "^RD" | cut -c1-40
$PY code/acc50.py 2>&1 | grep -v -i warn
echo FIN
