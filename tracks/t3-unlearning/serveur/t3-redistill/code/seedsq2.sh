#!/bin/bash
# usage: seedsq2.sh MAXJOBS "K GRAINE BS PAS" ...   file de redistillations ; saute un pli deja fini (out/NOM_logits.npz) ou deja en cours sur ce noeud
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=2; PY=~/eac/t3/.venv/bin/python; mkdir -p logs out tm/cache; MAX=$1; shift
one() { n=$1; shift
  for i in $(seq 1 40); do $PY code/rd.py --name $n --lite "$@" > logs/$n.log 2>&1 && return 0; grep -q -i -E "out of memory|ALLOC_FAILED|CUDA error" logs/$n.log || return 1; sleep 30; done; }
for cfg in "$@"; do set -- $cfg; k=$1; s=$2; bs=$3; st=$4
  for f in 0 1 2 3 4; do
    n=${PFX:-F}${k}s${s}_f$f
    [ -s out/${n}_logits.npz ] && continue
    pgrep -u adem -f -- "--name $n " > /dev/null && continue
    while [ $(jobs -rp | wc -l) -ge $MAX ]; do wait -n; done
    one $n --k $k --seed $s --fseed $s --hold f$f/5 --bs $bs --steps $st $RDX &
  done
done
wait; echo FIN_GRAINES
