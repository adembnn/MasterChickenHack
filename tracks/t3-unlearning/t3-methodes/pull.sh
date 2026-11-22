#!/bin/bash
# usage (sur le PC) : pull.sh NOM CIBLE   rapatrie le delta de h200e vers h200 puis y ecrit le fichier complet, le tableau de bord et la validation
n=$1; t=$2
ssh h200e "cd ~/eac/t3/t3-methodes && tar cf - deltas/$n.pt diag/row_$n.md diag/row_${n}ab.md 2>/dev/null" 2>/dev/null | ssh h200 "cd ~/eac/t3/t3-methodes && mkdir -p e && tar xf - -C e && mv e/deltas/$n.pt deltas/ && cp e/diag/row_${n}ab.md diag/ 2>/dev/null; echo 'saved $n (h200e)' > logs/$n.log; rm -f diag/row_$n.md; (OMP_NUM_THREADS=4 nohup code/finish.sh $n $t > /dev/null 2>&1 < /dev/null &)" 2>/dev/null
echo "pull $n lance"
