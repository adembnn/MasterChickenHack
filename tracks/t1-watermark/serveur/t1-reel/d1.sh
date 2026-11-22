#!/bin/bash
# h200d : TrustMark xhp4 (canaux passe-haut, réserve) ; PixelSeal x16 et x17 (généralistes, pleines)
cd ~/eac/t1; R=t1-reel
bash $R/c2.sh trustmark _c0,_c1,_c2,_c3 16000 xhp4 "--s1 1 --bs 16 --aux 1.0 --seed 3 --hp 2" "" &
bash $R/c2.sh pixelseal _w4,_w5 20000 x16 "--s1 1 --bs 16 --aux 1.0 --seed 17" "" &
bash $R/c2.sh pixelseal _w4,_w5 20000 x17 "--s1 1 --bs 16 --aux 1.0 --seed 18 --hp 2" "" &
wait; echo FINI
