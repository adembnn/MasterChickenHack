#!/bin/bash
# h200d : TrustMark g_c4 = g_c1 + tête d'appoint qui prédit le résidu signé (expérience X3 du critique), à pas égal (8000). Avant : essai à blanc, et moyenne sur 8 symétries pour f_s1.
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=2; V=/home/hermes/eac/kit/.venv/bin/python; PY="$V t1-reel/reel7.py"; R=t1-reel
C="--method trustmark --suffix _c0,_c1,_c2,_c3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --pshift 0.15 --pblob 0.15 --s1 1 --bs 16"
$PY train $C --tag smk4 --steps 10 --evalevery 10 --ckevery 0 --aux 1.0 2>&1 | grep -v -i warn | tail -2 | cut -c1-150; rm -rf $R/runs/smk4
(setsid nohup $PY train $C --tag g_c4 --steps 8000 --aux 1.0 > $R/logs/g_c4_trustmark.log 2>&1 < /dev/null &)
$V t1-reel/predict.py --method trustmark --tags f_s1 --train 1 --tta 8 --out t1-reel/verif_f_s1_tta8 2>&1 | grep -E "réserve|écrit" | cut -c1-160
$V t1-reel/predict.py --method trustmark --tags f_s1 --train 1 --tta 2 --out t1-reel/verif_f_s1_tta2 2>&1 | grep -E "réserve" | cut -c1-160
echo FINI
