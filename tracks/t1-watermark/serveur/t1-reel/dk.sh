#!/bin/bash
# h200d : on ne garde que TrustMark g_s1 ; arrêt des chaînes (sinon elles enchaîneraient l'affinage) puis des variantes écartées ; essai à blanc du tronc décodeur PixelSeal
pkill -f "bash d1.sh"; sleep 1
for t in "tag g_sts1" "tag g_lov" "tag g_r50" "method stegastamp" "tag g_st "; do pkill -f "reel5.py train.*$t"; done; sleep 2
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=3; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel5.py"; R=t1-reel
$PY train --method pixelseal --suffix _p --lazy 1 --fold -1 --nreal 0 --tag smkd --steps 30 --evalevery 15 --test 0 --arch dec 2>&1 | grep -v -i -E "warn|Model loaded" | tail -3 | cut -c1-300
rm -rf $R/runs/smkd
ps -eo etimes,cmd | grep "reel5.py train" | grep -v grep | cut -c1-120; nvidia-smi --query-gpu=memory.used --format=csv,noheader
