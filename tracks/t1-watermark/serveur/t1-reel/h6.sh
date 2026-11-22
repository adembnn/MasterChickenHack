#!/bin/bash
# h200h : la tige à filtres passe-haut appris lit deux fois mieux les petits rectangles PixelSeal (partielles 0,719 contre 0,701).
#  x7st : un troisième modèle PixelSeal de ce type (autres exemples, autre graine), 20 000 pas
#  dst  : la même tige sur TrustMark (question : gagne-t-elle aussi sur les partielles TrustMark ? témoins f_x1 / f_c5, partielles 0,775 à 0,783 en CV)
cd ~/eac/t1; R=t1-reel
bash $R/c2.sh pixelseal _w6,_w7 20000 x7st "--s1 1 --bs 16 --aux 1.0 --seed 6 --st 1 --hp 2" "" &
bash $R/c2.sh trustmark _d0,_d1,_d2,_d3 16000 dst "--s1 1 --bs 16 --aux 1.0 --seed 0 --st 1 --hp 2" "" &
wait; echo FINI
