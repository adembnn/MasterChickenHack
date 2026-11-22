#!/bin/bash
# h200d, PixelSeal, spécialiste des pleines (pendant de xsp) : 90 % d'exemples fabriqués pleins (--pfull 0.9), affinage sur les vraies pleines seulement. Part dès que x5hp a fini son pré-entraînement.
#  Témoin : pleines de f_x2 (CV 0,706 / réserve 0,710) et de la moyenne livrée (0,721 / 0,724).
cd ~/eac/t1; R=t1-reel
until [ -f $R/runs/g_x5hp/pixelseal/fold-1/done ]; do sleep 30; done
bash $R/c3.sh pixelseal _w4,_w5 16000 xfu "--s1 1 --bs 16 --aux 1.0 --seed 10 --pfull 0.9" "--realpop full"
echo FINI
