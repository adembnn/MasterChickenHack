#!/bin/bash
# h200b : TrustMark grande échelle. 3 réserves d'exemples en plus, puis un pré-entraînement long sur fabriqué seul (réserve suivie tous les 1000 pas), points de contrôle.
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=4; R=t1-reel; G="--padlo 10 --padhi 16 --rpadlo 2 --rpadhi 6"
for k in 1 2 3; do python $R/reel4.py gen --method trustmark --suffix _g$k --gseed $k $G > $R/logs/gen_g${k}_trustmark.log 2>&1; done
python $R/reel4.py train --method trustmark --suffix _p,_g1,_g2,_g3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --tag g_r34 --steps 20000 > $R/logs/g_r34_trustmark.log 2>&1
echo FINI
