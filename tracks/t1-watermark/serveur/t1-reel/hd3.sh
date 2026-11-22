#!/bin/bash
# h200d : (1) TrustMark g_c2 = g_c1 + greffes d'objets réalistes (vrais objets SAM d'une image filigranée sur sa toile, collés ailleurs avec déplacement), à pas égal avec g_c1 ;
#         (2) PixelSeal sur une toile plus large (s jusqu'à 100, mesure de t1-residus) : tronc décodeur g_decW puis affinage 5 plis f_decW ; U-Net pleine résolution g_s1W
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
C="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --steps 8000 --pshift 0.15 --pblob 0.15"
$PY train --method trustmark --suffix _c0,_c1,_c2,_c3 $C --s1 1 --bs 16 --tag g_c2 --objsam $R/sam > $R/logs/g_c2_trustmark.log 2>&1 &
( $PY gen --method pixelseal --suffix _w0 --gseed 20 --ncoco 0 --nvar 3 --canvas real --smax 100 --rpadlo 2 --rpadhi 6 > $R/logs/gen_w0_pixelseal.log 2>&1
  $PY train --method pixelseal --suffix _w0 $C --s1 1 --bs 16 --tag g_s1W --objsam $R/sam > $R/logs/g_s1W_pixelseal.log 2>&1 &
  $PY train --method pixelseal --suffix _w0 $C --arch dec --tag g_decW --objsam $R/sam > $R/logs/g_decW_pixelseal.log 2>&1
  for f in 0 1 2 3 4; do $PY train --method pixelseal --suffix _w0 --lazy 1 --fold $f --tag f_decW --init g_decW --arch dec --nreal 6 --raug 0 --tta 1 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 --pshift 0.15 --pblob 0.15 --objsam $R/sam > $R/logs/f_decW_pixelseal_f$f.log 2>&1 & [ $((f % 3)) -eq 2 ] && wait; done; wait
  $PY agg --tag f_decW,f_deca,f_decW+f_deca --write 1 2>&1 | grep -E "^\| (tag|f_).*pixelseal" > $R/logs/agg_f_decW_pixelseal.txt ) &
wait; echo FINI
