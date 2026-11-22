#!/bin/bash
# Arrete tous les processus de t3-simulateur sur cette machine (file d'attente, entrainements, bancs).
D=$HOME/eac/t3/t3-simulateur
for p in $(pgrep -u $USER -x queue.sh); do kill $p; done
for p in $(pgrep -u $USER -x python; pgrep -u $USER -x python3); do
  [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$D" ] && kill $p
done
sleep 5
n=0; for p in $(pgrep -u $USER -x python; pgrep -u $USER -x python3); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$D" ] && n=$((n+1)); done
echo "processus t3-simulateur restants : $n ; runner : $(pgrep -u $USER -x queue.sh | wc -l) ; GPU libre : $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ; UTC $(date -u +%H:%M)"
rm -f $D/queue.txt $D/queue.new
grep -h -v -E "Warn|xb = u8|for k, u8|^reserv|^cache|^prep|^modele" $D/logs/bench_B5.log 2>/dev/null | cut -c1-135 | tail -16
