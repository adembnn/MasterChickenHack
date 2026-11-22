#!/bin/bash
# h200h, P5 étape 1 bis : même test de détectabilité locale, quatre fois plus long (12 000 pas) et sur deux réserves d'exemples (6000 rectangles), pour savoir si 0,74 d'AUC était un manque d'entraînement
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib:$HOME/eac/t1/t1-reel T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python"; R=t1-reel
[ -f $R/data_pixelseal_w7/done ] || $PY $R/reel11.py gen --method pixelseal --suffix _w7 --gseed 27 --ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6 --smax 100 > $R/logs/gen_w7_pixelseal.log 2>&1
$PY $R/p5.py pixelseal _w6,_w7 12000 > $R/logs/p5b_pixelseal.log 2>&1; echo FINI
