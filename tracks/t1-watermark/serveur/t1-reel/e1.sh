#!/bin/bash
# h200e : PixelSeal sur la toile large : (1) U-Net pleine résolution + résidu, autre graine, 20 000 pas ; (2) tronc décodeur, 12 000 pas ; chacun suivi de son affinage 5 plis
cd ~/eac/t1; R=t1-reel
bash $R/c1.sh pixelseal _w0,_w1 20000 x2 "--s1 1 --bs 16 --aux 1.0 --seed 1" "" &
bash $R/c1.sh pixelseal _w0,_w1 12000 dx "--arch dec" "--raug 0 --tta 1" &
wait
