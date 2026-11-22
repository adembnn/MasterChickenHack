#!/bin/bash
# h200b : g_s1S = g_s1 avec des trous fabriqués fidèles aux vrais (formes réelles non décalées 85 % du temps, 15 % de taches aléatoires au lieu de 30 %), à pas égal (8000)
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=3; R=t1-reel
python $R/reel6.py train --method trustmark --suffix _p,_g1,_g2,_g3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --s1 1 --bs 16 --steps 8000 --tag g_s1S --pshift 0.15 --pblob 0.15 > $R/logs/g_s1S_trustmark.log 2>&1
echo FINI
