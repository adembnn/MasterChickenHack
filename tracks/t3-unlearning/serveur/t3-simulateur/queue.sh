#!/bin/bash
# File d'attente GPU : lance chaque ligne de queue.txt ("besoin_Mo|nom|commande") des que la carte a la place,
# sans depasser CAP Mo pour l'ensemble des processus de ce dossier. Une ligne lancee est retiree de queue.txt.
cd ~/eac/t3/t3-simulateur; source ../.venv/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CAP=${CAP:-20000}; MARGE=2500
mine() { t=0; for p in $(pgrep -u $USER -f "python (train|methods|robust)\.py"); do
           [ "$(readlink /proc/$p/cwd)" = "$PWD" ] || continue
           m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$p '$1==p{print $2}'); t=$((t + ${m:-0})); done; echo $t; }
while [ -s queue.txt ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); used=$(mine); n=0
  while IFS='|' read -r need name cmd; do
    n=$((n + 1)); [ -z "$need" ] && continue
    if [ $((free - MARGE)) -ge "$need" ] && [ $((used + need)) -le "$CAP" ]; then
      echo "$(date -u +%H:%M:%S) lance $name (libre $free Mo, a moi $used Mo)" >> logs/queue.log
      nohup bash -c "$cmd" > logs/$name.log 2>&1 &
      sed -i "${n}d" queue.txt; sleep 75; break
    fi
  done < queue.txt
  sleep 15
done
echo "$(date -u +%H:%M:%S) file vide" >> logs/queue.log
