#!/bin/bash
# h200h : TrustMark, même recette mais encodeur ResNet-50 (question : un réseau plus gros voit-il mieux les petites zones ?), 16 000 pas, seul sur la carte ; puis affinage 5 plis
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=4; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel7.py"; R=t1-reel
$PY cache
G="--ncoco 0 --nvar 3 --canvas real --rpadlo 2 --rpadhi 6"
for k in 0 1 2 3; do $PY gen --method trustmark --suffix _d$k --gseed 3$k $G > $R/logs/gen_d${k}_trustmark.log 2>&1 & done; wait
bash $R/c1.sh trustmark _d0,_d1,_d2,_d3 16000 r50 "--s1 1 --bs 16 --aux 1.0 --enc resnet50 --seed 2" ""
