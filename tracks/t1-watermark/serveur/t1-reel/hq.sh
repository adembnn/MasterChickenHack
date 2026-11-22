#!/bin/bash
# usage: hq.sh fichier_de_commandes nb_processus -> file d'attente : nb processus prennent les lignes une à une (reprise possible : les entraînements finis sont sautés)
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=3; R=t1-reel; CMDS=$1; PER=${2:-5}; ID=$(basename $CMDS .txt).$$
echo 0 > $R/logs/next.$ID
worker() {
  while true; do
    i=$(flock $R/logs/next.$ID.lock bash -c "i=\$(cat $R/logs/next.$ID); echo \$((i+1)) > $R/logs/next.$ID; echo \$i")
    [ "$i" -ge "$(wc -l < $CMDS)" ] && break
    cmd=$(sed -n "$((i+1))p" $CMDS); echo "$(date +%H:%M:%S) #$i $cmd" | cut -c1-160; bash -c "$cmd"
  done
}
for s in $(seq $PER); do worker & sleep 1; done; wait; echo FINI $CMDS
