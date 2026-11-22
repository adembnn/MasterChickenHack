#!/bin/bash
# tmhole ne finirait son pré-entraînement qu'à 11 h 15 : on l'arrête à son point de contrôle (8 000 pas, où il bat déjà le généraliste sur le trou) et on affine tout de suite, 5 plis
pkill -f "16000 tmhole"; pkill -f "tag g_tmhole "; pkill -f "tag g_tmhole$"; sleep 4
cd ~/eac/t1; export T1_THREADS=3; PY="$HOME/eac/t1/.venv/bin/python t1-reel/reel12.py"; R=t1-reel
ls -la $R/runs/g_tmhole/trustmark/fold-1/ | awk '{print $5, $9}'
for essai in 1 2 3; do for f in 0 1 2 3 4; do [ -f $R/runs/f_tmhole/trustmark/fold$f/done ] || { $PY train --method trustmark --suffix _c0,_c1,_c2,_c3 --fold $f --tag f_tmhole --init g_tmhole --lazy 1 --nreal 8 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --aux 1.0 --hp 2 --pfull 0.9 --realpop full > $R/logs/f_tmhole_trustmark_f$f.log 2>&1 & sleep 3; }; done; wait; done
$PY agg --tag f_tmhole --write 1 2>&1 | grep -E "^\| (tag|f_).*trustmark" > $R/logs/agg_f_tmhole_trustmark.txt; echo FINI
