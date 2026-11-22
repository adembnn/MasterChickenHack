#!/bin/bash
# h200d : TrustMark g_c3 = g_c2 + 40 % des trous des pleines alignés sur le contenu (régions SAM de l'image elle-même rendues propres), à pas égal (8000) avec g_c1 et g_c2
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
C="--method trustmark --suffix _c0,_c1,_c2,_c3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16 --objsam $R/sam --holesam 0.4"
$PY train $C --tag smk3 --steps 6 --evalevery 6 --ckevery 0 --pfull 1.0 --dump $R/lot_trous.png 2>&1 | grep -v -i warn | tail -1 | cut -c1-120; rm -rf $R/runs/smk3
$PY train $C --tag g_c3 --steps 8000 > $R/logs/g_c3_trustmark.log 2>&1
echo FINI
