#!/bin/bash
# h200e, PixelSeal, spécialiste des partielles : la moyenne par population sait à 100 % si une image est pleine ou partielle, donc le modèle des partielles n'a pas besoin de savoir lire les pleines.
#  xsp : tige passe-haut apprise, 90 % d'exemples fabriqués partiels (--pfull 0.1), la moitié des rectangles de petit côté < 64 px dont un quart de bandes ; affinage sur les vraies partielles seulement.
#  Témoin : f_st (partielles CV 0,719 / réserve 0,653), même nombre de pas (16 000).
cd ~/eac/t1; R=t1-reel
bash $R/c3.sh pixelseal _w0,_w1 16000 xsp "--s1 1 --bs 16 --aux 1.0 --seed 9 --st 1 --hp 2 --pfull 0.1 --rsmall 0.5 --rband 0.25" "--realpop part"
echo FINI
