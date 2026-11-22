#!/bin/bash
# h200h : TrustMark xhp5 (canaux passe-haut, autres exemples _d) ; PixelSeal x18 (généraliste) sur _w6,_w7
cd ~/eac/t1; R=t1-reel
bash $R/c2.sh trustmark _d0,_d1,_d2,_d3 16000 xhp5 "--s1 1 --bs 16 --aux 1.0 --seed 4 --hp 2" "" &
bash $R/c2.sh pixelseal _w6,_w7 20000 x18 "--s1 1 --bs 16 --aux 1.0 --seed 19" "" &
wait; echo FINI
