#!/bin/bash
# h200g : TrustMark xhp6 (canaux passe-haut) ; PixelSeal x19 (généraliste) et x20 (généraliste, --hp 2)
cd ~/eac/t1; R=t1-reel
bash $R/c2.sh trustmark _c0,_c1,_c2,_c3 16000 xhp6 "--s1 1 --bs 16 --aux 1.0 --seed 5 --hp 2" "" &
bash $R/c2.sh pixelseal _w0,_w1 20000 x19 "--s1 1 --bs 16 --aux 1.0 --seed 20" "" &
bash $R/c2.sh pixelseal _w0,_w1 20000 x20 "--s1 1 --bs 16 --aux 1.0 --seed 21 --hp 2" "" &
wait; echo FINI
