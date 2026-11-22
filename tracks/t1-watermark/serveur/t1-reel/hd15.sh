#!/bin/bash
# h200d, PixelSeal : deuxième modèle à tige passe-haut apprise (meilleur sur les partielles : 0,682 contre 0,668 à 12 000 pas), 20 000 pas, autre graine, autres exemples : spécialiste des partielles pour la moyenne par population
cd ~/eac/t1; R=t1-reel
until [ -f $R/data_pixelseal_w4/done ] && [ -f $R/data_pixelseal_w5/done ]; do sleep 20; done
bash $R/c2.sh pixelseal _w4,_w5 20000 x6st "--s1 1 --bs 16 --aux 1.0 --seed 5 --st 1 --hp 2" ""
echo FINI
