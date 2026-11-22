#!/bin/bash
# essai à blanc des nouveautés (canal de force, deux étages sans réduction) sur une petite réserve
cd ~/eac/t1; export PYTHONPATH=$HOME/eac/t1/pylib T1_THREADS=4; PY="/home/hermes/eac/kit/.venv/bin/python t1-reel/reel6.py"; R=t1-reel
$PY gen --method trustmark --suffix _smoke --small 1 --ncoco 0 --nrect 30 --padlo 10 --padhi 16 --rpadlo 2 --rpadhi 6 2>&1 | grep -v -i warn | tail -1
$PY emap --method trustmark --suffix _smoke 2>&1 | grep -v -i warn | tail -1
$PY train --method trustmark --suffix _smoke --lazy 1 --fold -1 --nreal 0 --tag smk --steps 20 --evalevery 20 --test 0 --s1 1 --bs 8 --echan 1 2>&1 | grep -v -i warn | tail -2 | cut -c1-260
$PY train --method trustmark --suffix _smoke --lazy 1 --fold 0 --nreal 3 --nmix 2 --tag smk2 --steps 12 --evalevery 12 --s1 2 --bs 8 --echan 1 --init smk 2>&1 | grep -v -i warn | tail -2 | cut -c1-260
ls $R/runs/smk2/trustmark/fold0/; rm -rf $R/runs/smk $R/runs/smk2 $R/data_trustmark_smoke
grep "\"step\"" $R/logs/g_s1_trustmark.log | tail -n 3 | cut -c1-330; grep "\"step\"" $R/logs/g_dec_pixelseal.log | tail -n 3 | cut -c1-330; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
