# file d'attente t3-poids : un calcul a la fois, attend 14 Go libres, relance jusqu'a 6 fois en cas d'echec (memoire GPU partagee)
cd ~/eac/t3/t3-poids/code; export T3ME=~/eac/t3/t3-poids OMP_NUM_THREADS=4 PY=~/eac/t3/.venv/bin/python DASH=~/eac/t3/t3-methodes/code/dash.py C=~/eac/t3/t3-poids/candidates
n=0
while IFS= read -r line; do
  n=$((n+1)); [ -z "$line" ] && continue; case "$line" in \#*) echo "$line"; continue;; esac
  for try in 1 2 3 4 5 6; do
    until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 129000 ]; do sleep 10; done
    echo ">>> [$n essai $try] $line"; if eval "$line"; then break; else echo "!!! echec [$n essai $try]"; sleep 30; fi
  done
done < "$1"
echo FINQ
