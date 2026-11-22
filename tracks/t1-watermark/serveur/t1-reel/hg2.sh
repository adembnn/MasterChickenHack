#!/bin/bash
# h200b (16 Go) : TrustMark sur fabriqué seul, 4 réserves d'exemples, 20 000 pas : U-Net de référence contre tige « stéganalyse », même nombre de pas.
cd ~/eac/t1 && source .venv/bin/activate; export T1_THREADS=4; R=t1-reel
until [ -f $R/data_trustmark_g3/done ]; do sleep 10; done
C="--method trustmark --suffix _p,_g1,_g2,_g3 --lazy 1 --fold -1 --nreal 0 --evalevery 1000 --ckevery 1000 --test 0 --steps 20000"
python $R/reel5.py train $C --tag g_r34 > $R/logs/g_r34_trustmark.log 2>&1 &
python $R/reel5.py train $C --tag g_st --st 1 --hp 2 > $R/logs/g_st_trustmark.log 2>&1 &
wait; echo FINI
