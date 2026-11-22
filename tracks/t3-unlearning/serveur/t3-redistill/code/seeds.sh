#!/bin/bash
# usage: seeds.sh "K GRAINE BS PAS" ...   lance 5 plis hors echantillon par configuration (mode leger), plis tires avec la graine
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=2; PY=~/eac/t3/.venv/bin/python; mkdir -p logs out tm/cache
one() { n=$1; shift
  for i in $(seq 1 40); do $PY code/rd.py --name $n --lite "$@" > logs/$n.log 2>&1 && return 0; grep -q -i -E "out of memory|ALLOC_FAILED|CUDA error" logs/$n.log || return 1; sleep 30; done; }
for cfg in "$@"; do set -- $cfg; k=$1; s=$2; bs=$3; st=$4
  for f in 0 1 2 3 4; do one F${k}s${s}_f$f --k $k --seed $s --fseed $s --hold f$f/5 --bs $bs --steps $st & done
done
wait; echo FIN_GRAINES
