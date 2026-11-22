#!/bin/bash
# h200d : PixelSeal, tronc = décodeur PixelSeal, fabriqué seul (recette corrigée), 12 000 pas, relevés et points de contrôle tous les 1000 pas
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=4; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel5.py"; R=t1-reel
$PY train --method pixelseal --suffix _p --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --tag g_dec --arch dec --steps 12000 > $R/logs/g_dec_pixelseal.log 2>&1
echo FINI
