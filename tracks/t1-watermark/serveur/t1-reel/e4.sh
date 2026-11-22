#!/bin/bash
# h200e, PixelSeal, deuxième spécialiste des pleines : 90 % d'exemples pleins ET 40 % de petits trous (0,5 à 4 % de l'image) : les pleines dont le trou fait moins de 8 % portent 36 % de la perte restante.
cd ~/eac/t1; bash t1-reel/c3.sh pixelseal _w0,_w1 16000 xfh "--s1 1 --bs 16 --aux 1.0 --seed 11 --pfull 0.9 --hsmall 0.4" "--realpop full"; echo FINI
