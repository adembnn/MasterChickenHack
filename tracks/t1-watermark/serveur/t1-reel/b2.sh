#!/bin/bash
# h200b : (1) TrustMark, deuxième modèle à canaux passe-haut (--hp 2, autre graine) : f_x1hp donne réserve 0,746 seul et +0,018 en réserve dans la moyenne (IC95 hors de 0) ; (2) PixelSeal, généraliste de plus pour les pleines (recette f_x2, autre graine)
cd ~/eac/t1; R=t1-reel
bash $R/c2.sh trustmark _c0,_c1,_c2,_c3 16000 xhp2 "--s1 1 --bs 16 --aux 1.0 --seed 1 --hp 2" "" &
bash $R/c2.sh pixelseal _w8,_w9 20000 x11 "--s1 1 --bs 16 --aux 1.0 --seed 12" "" &
wait; echo FINI
