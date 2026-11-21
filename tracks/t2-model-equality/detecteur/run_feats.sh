#!/bin/bash
# Relit en parallèle (3 à la fois) les mondes complets (.done) pas encore relus ; verrou par monde.
cd ~/eac-t2/detecteur
one() { j=$1; n=$(basename $j .jsonl); fam=${n%%_*}
  [ -e feats/$n.npz ] && return; mkdir feats/$n.lock 2>/dev/null || return
  .venv/bin/python feats.py --family $fam --inp $j --out feats 2>&1 | grep -v Loading; rmdir feats/$n.lock; }
export -f one
while true; do
  ls worlds/*.jsonl.done 2>/dev/null | sed 's/.done$//' | xargs -P 2 -I{} bash -c 'one {}'
  sleep 30
done
