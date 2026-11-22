#!/bin/bash
# h200b : arrêt de g_s1W (perte sur le trou dès le départ : très en retard à 1000 pas, la question sera reposée à l'affinage) ;
# PixelSeal dédié pleines, dès que la réserve d'exemples est prête : tronc décodeur (g_decF) et U-Net en pleine résolution (g_s1F)
pkill -f "reel6.py train.*tag g_s1W"
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=3; R=t1-reel; PY="python $R/reel6.py"
until [ -f $R/data_pixelseal_p/done ]; do sleep 15; done
C="--method pixelseal --suffix _p --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --steps 8000 --pfull 1.0 --pshift 0.15 --pblob 0.15"
$PY train $C --tag g_decF --arch dec > $R/logs/g_decF_pixelseal.log 2>&1 &
$PY train $C --tag g_s1F --s1 1 --bs 16 > $R/logs/g_s1F_pixelseal.log 2>&1 &
wait; echo FINI
