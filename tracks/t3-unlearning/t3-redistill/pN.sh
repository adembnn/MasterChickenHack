#!/bin/bash
# usage: pN.sh NOM_ETIQUETTE "URL_BASE graine graine ..." ["URL_BASE graine ..."]
# (h200c) attend que tous les plis des graines demandees soient telechargeables (serveurs http des noeuds de redistillation), les rassemble dans oof/,
# refait la moyenne de classe a T=2 sur TOUTES les partitions presentes, fabrique les cibles RDq<N> (rang de la base) et RDq<N>p (rang moyen), et lance les deux fichiers (graine 1).
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm; PY=~/eac/t3/.venv/bin/python; POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"; T=$HOME/eac/t3/t3-redistill/targets; tag=$1; shift
SPECS=("$@")
while :; do miss=0
  for spec in "${SPECS[@]}"; do read -r url seeds <<< "$spec"
    for s in $seeds; do for f in 0 1 2 3 4; do n=F4s${s}_f${f}_logits.npz
      [ -s oof/$n ] && continue
      wget -q -T 20 $url/$n -O oof/$n.part 2>/dev/null && mv oof/$n.part oof/$n || { rm -f oof/$n.part; miss=$((miss + 1)); }
    done; done
  done
  [ $miss -eq 0 ] && break; echo "$(date +%H:%M) manquants : $miss"; sleep 60
done
$PY code/oofavg.py 4 2.0 2>&1 | grep -v -i warn | grep "redistillations completes\|MOYENNE\|ecrit\|incomplete"
N=$(ls $T | grep -o "redistill_k4m[0-9]*_tau2.0.npy" | sed "s/redistill_k4m//;s/_tau2.0.npy//" | sort -n | tail -1); echo "partitions : $N"
$PY tmcode2/mk_tgt.py RDq${N}_70_l915 --pool $POOL --f 0.70 --llo 9 --lhi 15.5 --rank ext:$T/redistill_k4m3.npy --wsrc ext:$T/redistill_k4m${N}_tau2.0.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-24
$PY tmcode2/mk_tgt.py RDq${N}p_70_l915 --pool $POOL --f 0.70 --llo 9 --lhi 15.5 --rank ext:$T/redistill_k4m${N}.npy --wsrc ext:$T/redistill_k4m${N}_tau2.0.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-24
$PY code/shape.py RDq_70_l915 RDq${N}_70_l915 RDq${N}p_70_l915 2>&1 | grep -v -i warn
(nohup code/cand2.sh PG_RDq${N}_70_l915c_s1 RDq${N}_70_l915 --seed 1 > logs/c11_q$N.log 2>&1 < /dev/null &)
(nohup code/cand2.sh PG_RDq${N}p_70_l915c_s1 RDq${N}p_70_l915 --seed 1 > logs/c11_q${N}p.log 2>&1 < /dev/null &)
echo "LANCE_$N $(date +%H:%M)"
