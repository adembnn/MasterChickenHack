#!/bin/bash
# h200h -> PixelSeal : une réserve d'exemples, puis le test de détectabilité locale des petits rectangles (P5, étape 1)
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib:$HOME/eac/t1/t1-reel T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python"; R=t1-reel
[ -f $R/data_pixelseal_w6/done ] || $PY $R/reel11.py gen --method pixelseal --suffix _w6 --gseed 26 --ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6 --smax 100 > $R/logs/gen_w6_pixelseal.log 2>&1
$PY $R/p5.py pixelseal _w6 3000 > $R/logs/p5_pixelseal.log 2>&1; echo FINI
