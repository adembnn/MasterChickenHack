#!/bin/bash
# h200b : relance après la perte de h200d. TrustMark g_s1 (pleine résolution, 8000 pas, métrique « trou »), puis affinage 5 plis ; PixelSeal : réserve d'exemples, tronc décodeur, affinage 5 plis.
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=3; R=t1-reel; PY="python $R/reel6.py"
C="--lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0"
( $PY train --method trustmark --suffix _p,_g1,_g2,_g3 $C --tag g_s1 --s1 1 --bs 16 --steps 8000 > $R/logs/g_s1_trustmark.log 2>&1
  for f in 0 1 2 3 4; do
    $PY train --method trustmark --suffix _p,_g1,_g2,_g3 --lazy 1 --fold $f --tag f_s1 --init g_s1 --s1 1 --bs 16 --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 > $R/logs/f_s1_trustmark_f$f.log 2>&1 &
    [ $((f % 2)) -eq 1 ] && wait
  done; wait; $PY agg --tag f_s1 --write 1 2>&1 | grep -E "^\| (tag|f_)" > $R/logs/agg_f_s1_trustmark.txt ) &
( [ -f $R/data_pixelseal_p/done ] || $PY gen --method pixelseal --suffix _p --padlo 10 --padhi 16 --rpadlo 2 --rpadhi 6 > $R/logs/gen_p_pixelseal.log 2>&1
  $PY train --method pixelseal --suffix _p $C --tag g_dec --arch dec --steps 8000 > $R/logs/g_dec_pixelseal.log 2>&1
  for f in 0 1 2 3 4; do
    $PY train --method pixelseal --suffix _p --lazy 1 --fold $f --tag f_dec --init g_dec --arch dec --nreal 6 --raug 0 --tta 1 --steps 1000 --lr 1e-4 --warm 0.1 --ckevery 250 --evalevery 250 > $R/logs/f_dec_pixelseal_f$f.log 2>&1 &
    [ $((f % 3)) -eq 2 ] && wait
  done; wait; $PY agg --tag f_dec --write 1 2>&1 | grep -E "^\| (tag|f_)" > $R/logs/agg_f_dec_pixelseal.txt ) &
wait; echo FINI
