#!/bin/bash
# h200d : suite des relevés à pas égal. Arrêt de g_c2 (greffes réalistes : aucun gain sur les partielles). La prédiction du résidu (g_c4) gagne ~1000 pas d'avance :
#  - TrustMark : affinage 5 plis de g_c1 (toile réelle) dès sa fin -> f_c1 ; de g_c4 (toile réelle + résidu) dès sa fin -> f_c4 ; g_c5 = g_c4 sur 16 000 pas
#  - StegaStamp : g_c4 (pleine résolution + toile réelle + résidu), 8000 pas, puis affinage 5 plis f_c4
pkill -f "reel6.py train.*tag g_c2"
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
P="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16"; F="--lazy 1 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16"
TM="--method trustmark --suffix _c0,_c1,_c2,_c3"; SS="--method stegastamp --suffix _c0"
ft() {  # ft methode etiquette_affinee etiquette_depart "donnees" "options"
  until [ -f $R/runs/$3/$1/fold-1/done ]; do sleep 30; done
  for f in 0 1 2 3 4; do $PY train $4 --fold $f --tag $2 --init $3 $F $5 > $R/logs/${2}_${1}_f$f.log 2>&1 & [ $((f % 3)) -eq 2 ] && wait; done; wait
  $PY agg --tag $2 --write 1 2>&1 | grep -E "^\| (tag|f_).*$1" > $R/logs/agg_${2}_$1.txt
}
ft trustmark f_c1 g_c1 "$TM" "" &
ft trustmark f_c4 g_c4 "$TM" "--aux 1.0" &
$PY train $TM $P --tag g_c5 --steps 16000 --aux 1.0 > $R/logs/g_c5_trustmark.log 2>&1 &
( $PY train $SS $P --tag g_c4 --steps 8000 --aux 1.0 > $R/logs/g_c4_stegastamp.log 2>&1; ft stegastamp f_c4 g_c4 "$SS" "--aux 1.0" ) &
wait; echo FINI
