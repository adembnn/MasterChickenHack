#!/bin/bash
# h200c : moyenne de toutes les partitions disponibles a 4 blocs, puis 4 fichiers sur la base du 0,3888 / 0,3942 (un ou deux facteurs changes, voir noms)
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm; PY=~/eac/t3/.venv/bin/python; POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"; T=$HOME/eac/t3/t3-redistill/targets
$PY code/oofavg.py 4 2.4 2>&1 | grep -v -i warn
TAG=$(ls $T | grep -o "redistill_k4m[0-9]*_tau2.4.npy" | sed "s/redistill_//;s/_tau2.4.npy//" | sort -t m -k2 -n | tail -1); echo "etiquette $TAG"
$PY code/rankp.py $TAG 2>&1 | grep -v -i warn
mk() { $PY tmcode2/mk_tgt.py "$@" 2>&1 | grep "^RD" | cut -c1-28; }
mk RDp_70_l1016 --pool $POOL --f 0.70 --llo 10 --lhi 16.5 --rank ext:$T/redistill_$TAG.npy --wsrc ext:$T/redistill_${TAG}_tau2.4.npy --wext 1.0
mk RDpc_70_l1016 --pool $POOL --f 0.70 --llo 10 --lhi 16.5 --rank ext:$T/redistill_k4m3.npy --wsrc ext:$T/redistill_${TAG}_tau2.4.npy --wext 1.0
mk RDpr_70_l1016 --pool $POOL --f 0.70 --llo 10 --lhi 16.5 --rank ext:$T/redistill_$TAG.npy --wsrc ext:$T/redistill_tau2.4.npy --wext 1.0
mk RDpv_70_l1016 --pool $POOL --f 0.70 --llo 10 --lhi 16.5 --rank ext:$T/redistill_rang_${TAG}v.npy --wsrc ext:$T/redistill_${TAG}_tau2.4.npy --wext 1.0
$PY code/shape.py RDt24r_70_l1016 RDp_70_l1016 RDpc_70_l1016 RDpr_70_l1016 RDpv_70_l1016 2>&1 | grep -v -i warn
echo FIN_CIBLES
code/cand.sh RDp_70_l1016 RDpc_70_l1016 RDpr_70_l1016 RDpv_70_l1016
