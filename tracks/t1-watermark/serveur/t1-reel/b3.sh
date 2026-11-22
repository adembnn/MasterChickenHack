#!/bin/bash
# h200b, remplissage : PixelSeal x14st (tige passe-haut, partielles) et x15 (généraliste, pleines), autres graines
cd ~/eac/t1; R=t1-reel
bash $R/c2.sh pixelseal _w8,_w9 16000 x14st "--s1 1 --bs 16 --aux 1.0 --seed 15 --st 1 --hp 2" "" &
bash $R/c2.sh pixelseal _w8,_w9 20000 x15 "--s1 1 --bs 16 --aux 1.0 --seed 16" "" &
wait; echo FINI
